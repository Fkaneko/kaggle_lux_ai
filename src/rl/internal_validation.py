import copy
import math
import random
import sys
import time
from functools import partial  # pip install functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from gym import spaces
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.game.actions import (
    MoveAction,
    ResearchAction,
    SpawnCartAction,
    SpawnCityAction,
    SpawnWorkerAction,
)
from luxai2021.game.constants import Constants, LuxMatchConfigs_Default
from luxai2021.game.game import Game
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.match_controller import GameStepFailedException
from tqdm import tqdm

from src.dataset.dataset import (
    LuxDataset,
    make_input_from_game_state,
    order_unit,
    update_active_unit_on_input,
)
from src.rl.luxgym_to_handyrl import (
    Player,
    UnitOrder,
    default_city_action,
    get_can_act_units,
)
from src.rl.model import PolicyValueNet
from src.rl.utils import (
    StateHist,
    check_action_plan,
    check_is_center_action,
    crop_state,
    decide_worker_gen_place,
    get_act_cities_map,
    get_resource_distribution,
    get_unit_sequence_obs,
    pred_with_onnx,
    to_numpy,
)

ONE_CYCLE_LENGTH = (
    GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
    + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
)


def in_city(pos, game_state, team):
    try:
        city = game_state.map.get_cell_by_pos(pos).city_tile
        return city is not None and city.team == team
    except:
        return False


def is_day(game: Game) -> bool:
    return (
        game.state["turn"] % ONE_CYCLE_LENGTH
        < GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
    )


class ImitationAgent(AgentWithModel):
    def __init__(
        self, mode="train", model=None, is_xy_order: bool = False, is_cuda: bool = False
    ) -> None:
        """
        Implements an agent opponent
        """
        if is_cuda:
            model = model.cuda()
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.action_space = spaces.Discrete(len(self.actions_units))

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 32, 32), dtype=np.float16
        )
        self.is_xy_order = is_xy_order
        self.is_cuda = is_cuda

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        if is_new_turn:
            # map related generation
            self.obs_map, self.unit_stack_map = make_input_from_game_state(
                game=game,
                active_unit_id=None,
                player_index=team,
                num_state_features=self.observation_space.shape[0],
                input_size=tuple(self.observation_space.shape[1:]),
            )

        # add active unit
        b_active = self.obs_map.copy()
        b_active = update_active_unit_on_input(
            b=b_active,
            game=game,
            unit_id=unit.id,
            player_index=team,
            input_size=tuple(self.observation_space.shape[1:]),
            unit_stack_map=self.unit_stack_map,
        )
        # only for kaggle public notebook compatibility
        if self.is_xy_order:
            b_active = b_active.transpose((0, 2, 1))

        self.obs = b_active

        return self.obs

    def action_code_to_action(
        self, action_code, game, unit=None, city_tile=None, team=None
    ):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile != None:
                action = self.actions_cities[action_code % len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y,
                )
            else:
                action = self.actions_units[action_code % len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y,
                )

            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_finished:
            if game.get_winning_team() == self.team:
                return 1  # Win!
            else:
                return -1  # Loss

        return 0

    def make_batch_obs(
        self, game: Game, units: list, new_turn: bool = True
    ) -> torch.Tensor:

        batched_obs = []
        action_units = []
        for unit in units:
            if unit.can_act() and (
                is_day(game=game)
                or not in_city(unit.pos, game_state=game, team=unit.team)
            ):
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                if new_turn:
                    new_turn = False
                batched_obs.append(obs)
                action_units.append(unit)
        if len(batched_obs) == 0:
            return None, None, new_turn
        else:
            batched_obs = np.stack(batched_obs)
            return (
                torch.from_numpy(batched_obs),
                action_units,
                new_turn,
            )

    def infer_with_batch(
        self, actions: list, model, game: Game, units: list, new_turn: bool = True
    ) -> list:
        batched_obs, action_units, new_turn = self.make_batch_obs(
            game=game, units=units, new_turn=new_turn
        )
        if batched_obs is None:
            # there are no can_cat units
            return actions, new_turn
        else:
            with torch.no_grad():
                if self.is_cuda:
                    batched_obs = batched_obs.cuda()
                else:
                    batched_obs = batched_obs.type_as(next(self.model.parameters()))
                outputs = self.model(batched_obs)
                if isinstance(outputs, dict):
                    outputs = outputs["outputs"]
                action_codes = torch.argmax(outputs, dim=-1).cpu()

            for action_unit, action_code in zip(action_units, action_codes):
                actions.append(
                    self.action_code_to_action(
                        action_code.numpy(),
                        game=game,
                        unit=action_unit,
                        city_tile=None,
                        team=action_unit.team,
                    )
                )

            return actions, new_turn

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        actions, new_turn = self.infer_with_batch(
            actions=actions, model=self.model, game=game, units=units, new_turn=new_turn
        )
        # for unit in units:
        #     if unit.can_act():
        #         obs = self.get_observation(game, unit, None, unit.team, new_turn)
        #         # IMPORTANT: You can change deterministic=True to disable randomness in model inference. Generally,
        #         # I've found the agents get stuck sometimes if they are fully deterministic.
        #         # action_code, _states = self.model.predict(obs, deterministic=False)
        #         with torch.no_grad():
        #             obs = torch.from_numpy(obs).unsqueeze(0)
        #             if self.is_cuda:
        #                 obs = obs.cuda()
        #                 action_code = torch.argmax(self.model(obs)).cpu()
        #             else:
        #                 action_code = torch.argmax(self.model(obs))

        #         if action_code is not None:
        #             actions.append(
        #                 self.action_code_to_action(
        #                     action_code.numpy(),
        #                     game=game,
        #                     unit=unit,
        #                     city_tile=None,
        #                     team=unit.team,
        #                 )
        #             )
        #         new_turn = False

        unit_count = len(units)

        # Inference the model per-city
        cities = game.cities.values()
        player_cities = [city for city in cities if city.team == team]
        player = Player(
            research_points=game.state["teamStates"][team]["researchPoints"],
            city_tile_count=sum([len(city.city_cells) for city in player_cities]),
        )
        for city in player_cities:
            for cell in city.city_cells:
                city_tile = cell.city_tile
                if city_tile.can_act():
                    # obs = self.get_observation(
                    #     game, None, city_tile, city.team, new_turn
                    # )
                    if unit_count < player.city_tile_count:
                        # actions.append(city_tile.build_worker())
                        actions.append(
                            SpawnWorkerAction(
                                game=game,
                                unit_id=None,
                                unit=None,
                                team=team,
                                x=city_tile.pos.x,
                                y=city_tile.pos.y,
                            )
                        )
                        unit_count += 1
                    elif not player.researched_uranium():
                        # actions.append(city_tile.research())
                        actions.append(
                            ResearchAction(
                                game=game,
                                team=team,
                                unit_id=None,
                                unit=None,
                                x=city_tile.pos.x,
                                y=city_tile.pos.y,
                            )
                        )
                        player.research_points += 1
                    # for the citytile there is no obs, so no new_turn
                    # new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print(
                "WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second."
                % time_taken,
                file=sys.stderr,
            )

        return actions


class ImageCaptionAgent(ImitationAgent):
    def __init__(self, mode="train", model=None) -> None:
        super().__init__(mode=mode, model=model, is_xy_order=False, is_cuda=False)
        self.unit_length = 128
        self.obs = {0: None, 1: None}
        self.unit_order = {
            0: UnitOrder(step=0, order_strings=[]),
            1: UnitOrder(step=0, order_strings=[]),
        }
        self.is_onnx_run = False
        if isinstance(model, ort.InferenceSession):
            self.is_onnx_run = True

        self.hist_folder = {0: StateHist(), 1: StateHist()}
        self.hist_folder[0].reset()
        self.hist_folder[1].reset()
        self.use_hist = False

    def game_start(self, game):
        self.hist_folder[0].reset()
        self.hist_folder[1].reset()
        self.obs = {0: None, 1: None}
        self.unit_order = {
            0: UnitOrder(step=0, order_strings=[]),
            1: UnitOrder(step=0, order_strings=[]),
        }

    def get_observation(self, game, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        player = team
        if is_new_turn:
            # map related generation
            self.obs_map, self.unit_stack_map = make_input_from_game_state(
                game=game,
                active_unit_id=None,
                player_index=team,
                num_state_features=self.observation_space.shape[0],
                input_size=tuple(self.observation_space.shape[1:]),
            )

        b_active = self.obs_map.copy()
        units = game.state["teamStates"][team]["units"]
        can_act_units = {
            unit_id: unit for unit_id, unit in units.items() if unit.can_act()
        }

        obs, unit_order = get_unit_sequence_obs(
            game=game,
            player=team,
            b_active=b_active,
            can_act_units=can_act_units,
            turn=game.state["turn"],
            unit_length=self.unit_length,
            action_dim=len(self.actions_units),
            input_dim=4,
        )
        self.obs[player] = obs
        self.unit_order[player] = unit_order
        hist_right, hist_left = self.hist_folder[team].get_hist()
        if self.use_hist:
            self.obs[player]["image"][:2] = hist_left
            self.obs[player]["image"] = np.concatenate(
                [self.obs[player]["image"], hist_right], axis=0
            )

        return self.obs[player]

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = {0: [], 1: []}
        new_turn = True

        obs_0 = self.get_observation(game, team, new_turn)
        map_size = (game.map.height, game.map.width)
        current_action_plan = np.ones(map_size, dtype=bool)

        action_logit = self.predict_from_obs(
            model=self.model, obs=obs_0, is_onnx_run=self.is_onnx_run
        )

        for seq_ind, unit_id in enumerate(self.unit_order[team].order_strings):
            if seq_ind >= self.unit_length:
                break
            unit = game.state["teamStates"][team]["units"][unit_id]
            action_code = np.argmax(action_logit[seq_ind], axis=-1)

            our_city = crop_state(obs_0["image"][8] > 0, game=game)

            is_center = check_is_center_action(action_code=action_code)
            current_action_plan, use_cooldown_as_center = check_action_plan(
                action_code=action_code,
                our_city=our_city,
                pos_x=unit.pos.x,
                pos_y=unit.pos.y,
                current_plan=current_action_plan,
                is_center=is_center,
            )
            if use_cooldown_as_center:
                actions[team].append(
                    MoveAction(
                        direction=Constants.DIRECTIONS.CENTER,
                        game=game,
                        team=team,
                        unit_id=unit.id,
                        unit=unit,
                        x=unit.pos.x,
                        y=unit.pos.y,
                    )
                )

            elif not is_center:
                actions[team].append(
                    self.action_code_to_action(
                        action_code,
                        game=game,
                        unit=unit,
                        city_tile=None,
                        team=team,
                    )
                )

        cities = game.cities.values()
        player_cities = [city for city in cities if city.team == team]
        city_tile_count = sum([len(city.city_cells) for city in player_cities])

        units = game.state["teamStates"][team]["units"].values()
        unit_count = len(units)

        player_tiles = None
        act_cities_map_max, posyx2tile = get_act_cities_map(
            player_cities=player_cities, game_map=game.map
        )
        if (len(posyx2tile) > 0) and (unit_count < city_tile_count):
            res_avg_map, resource_map = get_resource_distribution(
                b_active=obs_0["image"], game=game
            )
            places = decide_worker_gen_place(
                res_avg_map=res_avg_map,
                resource_map=resource_map,
                act_cities_map_max=act_cities_map_max,
                num_units=unit_count,
                num_city_tiles=city_tile_count,
            )
            if len(places) > 0:
                player_tiles = []
                for y, x in places:
                    player_tiles.append(posyx2tile[y][x])

        city_actions = default_city_action(
            game=game, team=team, player_tiles=player_tiles
        )

        actions[team] = actions[team] + city_actions

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print(
                "WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second."
                % time_taken,
                file=sys.stderr,
            )
        self.hist_folder[team].update(
            current_state=self.obs[team]["image"], env_step=game.state["turn"]
        )
        return actions[team]

    def predict_from_obs(
        self,
        model: Union[torch.nn.Module, ort.InferenceSession],
        obs: Dict[str, np.ndarray],
        is_onnx_run: bool = False,
    ):
        # Inference
        if is_onnx_run:
            action_logit = pred_with_onnx(model=model, obs=obs)

        else:
            obs_torch = {}
            param = next(self.model.parameters())
            for key in ["image", "input_sequence", "rule_mask"]:
                obs_torch[key] = torch.Tensor(obs[key]).unsqueeze(0).type_as(param)

            with torch.no_grad():
                if isinstance(self.model, PolicyValueNet):
                    out0 = self.model(obs_torch)
                else:
                    out0 = self.model(
                        obs_torch["image"],
                        aux_inputs=obs_torch["input_sequence"],
                    )

            action_logit = out0.get("outputs", None)
            if action_logit is None:
                action_logit = out0.get("policy")
            action_logit = (
                action_logit - (obs_torch["rule_mask"] == 0) * 1e32
            ).squeeze(0)
            action_logit = to_numpy(action_logit)
        return action_logit


def calc_match_score(win_results: np.ndarray, scores: np.ndarray):
    score_diff = scores[:, 0] - scores[:, 1]
    win_rate = (win_results == 0).sum() / len(win_results)
    print("win rate for team 0", win_rate)
    print("mean score diff [0] - [1]", score_diff.mean())
    return win_rate, score_diff.mean()


def vis_match_results(scores: List[List[int]]):
    nrows = 1
    ncols = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 6),
        sharey=False,
        sharex=False,
    )
    axes.plot(np.arange(scores.shape[0]), scores[:, 0], "-o", label="agent:0")
    axes.plot(np.arange(scores.shape[0]), scores[:, 1], "-o", label="agent:1")
    axes.set_xlabel("episode")
    axes.set_ylabel("scores")
    axes.grid()
    axes.legend()
    return fig, axes


def get_team2agent(env: LuxEnvironment) -> Dict[int, Tuple[str, int]]:
    return {
        env.match_controller.agents[0].team: ("player", 0),
        env.match_controller.agents[1].team: ("oppenent", 1),
    }


def calc_match_result(
    env: LuxEnvironment,
    is_normalize: bool = False,
):
    # including citytile, units and fuel generation count
    win_team = env.game.get_winning_team()

    # count city tiles
    team_city_tile_count = [0, 0]
    for city in env.game.cities.values():
        team_city_tile_count[city.team] += len(city.city_cells)

    if is_normalize:
        norm = calc_score_normalization(game=env.game)
        team_city_tile_count = [score * norm for score in team_city_tile_count]

    # units_count = [
    #     len(env.game.state["teamStates"][0]["units"]),
    #     len(env.game.state["teamStates"][1]["units"]),
    # ]

    # agent team assignment order is randomized

    team2agent = get_team2agent(env=env)

    win_agent = team2agent[win_team][1]

    agent_city_tile_count = [0, 0]
    for team_id, tile_count in enumerate(team_city_tile_count):
        agent_id = team2agent[team_id][1]
        agent_city_tile_count[agent_id] = tile_count

    return win_agent, agent_city_tile_count


# Create the two agents that will play eachother
# Create a default opponent agent that does nothing
# opponent = Agent()
# batch cuda 14s
# batch cpu 47s
#  cuda 40s
#  cpu 1min40s


def internal_match(
    player: AgentWithModel,
    opponent: AgentWithModel,
    num_episodes: int = 10,
    seed: int = 42,
    seeds: Optional[List[int]] = None,
    map_size: Optional[int] = None,
    replay_folder: Optional[str] = None,
    replay_stateful: bool = True,
    vis_results: bool = False,
):

    # Create a game environment
    configs = copy.deepcopy(LuxMatchConfigs_Default)
    # Inference
    win_results = []
    scores = []
    replay_filepaths = []
    for episode_idx in tqdm(range(num_episodes), total=num_episodes):
        if (seeds is not None) and (map_size is not None):
            print("fixed ssed", seed, map_size)
            configs["seed"] = seeds[episode_idx]
            configs["width"] = map_size
            configs["height"] = map_size
        else:
            configs["seed"] = seed + random.randint(0, 10000)
        win_result, score, replay_filepath = run_one_match(
            player=player,
            opponent=opponent,
            configs=configs,
            replay_folder=replay_folder,
            replay_stateful=replay_stateful,
            episode_idx=episode_idx,
        )
        win_results.append(win_result)
        scores.append(score)
        replay_filepaths.append(replay_filepath)

    scores = np.array(scores)
    win_results = np.array(win_results)

    win_rate, score_diff_mean = calc_match_score(win_results=win_results, scores=scores)

    if vis_results:
        vis_match_results(scores=scores)
    return {
        "win_rate": win_rate,
        "score_diff_mean": score_diff_mean,
        "scores": scores,
        "replay_filepaths": replay_filepaths,
    }


def run_one_match(
    player: AgentWithModel,
    opponent: AgentWithModel,
    configs: Dict[str, Any],
    replay_folder: Optional[str] = None,
    replay_stateful: bool = True,
    opponent_name: str = "imitation_baseline",
    episode_idx: int = 0,
) -> Tuple[int, List[int], str]:

    # agent team assignment order is randomized at __init__, Matchcontroller.reset()
    env = LuxEnvironment(
        configs=configs,
        learning_agent=player,
        opponent_agent=opponent,
        replay_folder=replay_folder,
    )
    is_game_error, replay_filepath = run_no_learn_with_env(
        env=env,
        replay_stateful=replay_stateful,
        opponent_name=opponent_name,
        episode_idx=episode_idx,
    )
    if not is_game_error:
        win_result, score = calc_match_result(env=env)
        replay_filepath = Path(replay_filepath)
        new_replay_filepath = Path(
            replay_filepath.parent,
            replay_filepath.stem + f'_score_{"-".join(map(str, score))}.json',
        )
        replay_filepath.rename(new_replay_filepath)

    return win_result, score, str(new_replay_filepath)


def run_no_learn_with_env(
    env: LuxEnvironment,
    replay_stateful: bool = True,
    opponent_name: str = "imitation_baseline",
    episode_idx: int = 0,
):
    """
    Steps until the environment is "done".
    Both agents have to be in inference mode
    """

    for agent in env.match_controller.agents:
        assert (
            agent.get_agent_type() == Constants.AGENT_TYPE.AGENT
        ), "Both agents must be in inference mode"

    env.current_step = 0
    env.last_observation_object = None

    # Reset game + map
    env.match_controller.reset(randomize_team_order=False)

    # force to use ordered team assignment
    env.match_controller.agents[0].set_team(0)
    env.match_controller.agents[1].set_team(1)
    env.match_controller.reset(reset_game=False, randomize_team_order=False)

    replay_filepath = None
    if env.replay_folder:
        # Tell the game to log replays
        env.game.start_replay_logging(
            stateful=replay_stateful,
            replay_folder=env.replay_folder,
            # replay_filename_prefix=env.replay_prefix,
            replay_filename_prefix=f'ep_{episode_idx}_seed_{env.game.configs["seed"]}_rand',
        )
        replay_filepath = env.game.replay.file

        team2agent = get_team2agent(env=env)
        agent_logging_name = [
            {"name": "learning_agent", "tournamentID": ""},
            {"name": opponent_name, "tournamentID": ""},
        ]
        for team_id in range(len(env.game.replay.data["teamDetails"])):
            agent_id = team2agent[team_id][1]
            env.game.replay.data["teamDetails"][team_id] = agent_logging_name[agent_id]

    # Running
    env.match_generator = env.match_controller.run_to_next_observation()
    try:
        next(env.match_generator)
    except StopIteration:
        # The game episode is done.
        is_game_error = False
        # print("Episode run finished successfully!")
    except GameStepFailedException:
        # Game step failed.
        is_game_error = True
        print("Episode run failed")

    return is_game_error, replay_filepath


def calc_score_normalization(game: Game):
    turn_consumed = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
    area = math.sqrt((game.map.height * game.map.width) / 32 ** 2)
    return turn_consumed * area


def load_baseline_model(path="../input/lux_ai_baseline_imitation_weight"):
    model = torch.jit.load(f"{path}/model.pth")
    model.eval()
    return model


def load_opponent(
    path="../input/lux_ai_baseline_imitation_weight",
):
    model = load_baseline_model(path=path)
    opponent = ImitationAgent(mode="pred", model=model, is_xy_order=True, is_cuda=False)
    return opponent


if __name__ == "__main__":
    player_model = load_baseline_model()
    opponent = load_opponent()

    with initialize(config_path="./src/config"):
        conf = compose(config_name="config")
        print(OmegaConf.to_yaml(conf))


    lit_model = LitModel(conf=conf)
    rl_model = PolicyValueNet(
        encoder=lit_model.model.encoder,
        decoder=lit_model.model.decoder,
    )

    rl_model = rl_model.eval()

    player = ImageCaptionAgent(
        mode="pred", model=rl_model, is_xy_order=True, is_cuda=False
    )

    res = internal_match(
        player,
        opponent,
        replay_folder="../working/replay_debug",
        replay_stateful=False,
        vis_results=True,
    )
    print(res)

    # action_code, _states = model.predict(obs, deterministic=False)
    # (obs, reward, is_game_over, state) = env.step(action_code)
