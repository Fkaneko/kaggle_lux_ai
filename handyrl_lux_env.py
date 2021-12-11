# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import copy
import itertools
import random
import time
import traceback
from functools import partial  # pip install functools
from pathlib import Path
from typing import Dict, List, NamedTuple, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from hydra import compose, initialize
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.env.lux_env import LuxEnvironment
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
from luxai2021.game.match_controller import GameStepFailedException, MatchController
from luxai2021.game.unit import Unit
from omegaconf import OmegaConf

# from ...environment import BaseEnvironment
from HandyRL.handyrl.environment import BaseEnvironment
from src.dataset.dataset import (
    LuxDataset,
    make_input_from_game_state,
    update_active_unit_on_input,
)
from src.modeling.pl_model import LitModel
from src.rl.internal_validation import calc_match_result
from src.rl.luxgym_to_handyrl import (
    MatchControllerHandyrl,
    Reward,
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
    generate_sequence,
    get_act_cities_map,
    get_resource_distribution,
    order_unit,
)


class Environment(BaseEnvironment):
    ACTION = ["NORTH", "SOUTH", "WEST", "EAST"]
    DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    NUM_AGENTS = 2

    ACTION = [
        partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
        partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
        partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
        partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
        SpawnCityAction,
        # partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),
    ]

    def __init__(self, args={}):
        super().__init__()

        # Create a game environment
        configs = LuxMatchConfigs_Default

        player = Agent()
        opponent = Agent()

        env = LuxEnvironment(
            configs=configs, learning_agent=player, opponent_agent=opponent
        )
        env.match_controller = MatchControllerHandyrl(
            env.game, agents=[player, opponent], replay_validate=None
        )

        # force to use ordered team assignment
        env.match_controller.agents[0].set_team(0)
        env.match_controller.agents[1].set_team(1)
        env.match_controller.reset(reset_game=False, randomize_team_order=False)

        self.env = env
        self.reset()

        self.actions_units = self.ACTION
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 32, 32), dtype=np.float16
        )
        self.win_agent_hist = {0: 0, 1: 0}
        self.unit_length = 64
        self.use_hist = False

    def reset(self, args={}):
        # manual env reset
        self.env.current_step = 0
        self.env.last_observation_object = None
        # Reset game + map
        self.env.match_controller.reset(randomize_team_order=False)

        if self.env.replay_folder:
            # Tell the game to log replays
            self.env.game.start_replay_logging(
                stateful=True,
                replay_folder=self.env.replay_folder,
                replay_filename_prefix=self.env.replay_prefix,
            )

        self.env.match_generator = self.env.match_controller.run_to_next_observation()
        (_, _, _, is_new_turn) = next(self.env.match_generator)
        self.update(game=None, last_actions=None)
        self.obs = {0: None, 1: None}
        self.unit_order = {
            0: UnitOrder(step=0, order_strings=[]),
            1: UnitOrder(step=0, order_strings=[]),
        }

        self.reward_holder = {0: Reward(team=0), 1: Reward(team=1)}

        self.hist_folder = {0: StateHist(), 1: StateHist()}
        self.hist_folder[0].reset()
        self.hist_folder[1].reset()

        assert is_new_turn

        self.env.is_game_over = False

    def update(self, game: Game, last_actions: list, reset: bool = False):
        # if reset:
        #     self.obs_list = []
        #     self.last_obs = None
        #     self.last_actions = None
        self.last_obs = copy.copy(game)
        self.last_actions = last_actions

    def check_is_continue(self):
        """
        Implements /src/logic.ts -> getResults()
        """
        # count city tiles
        city_tile_count = [0, 0]
        for city in self.env.game.cities.values():
            city_tile_count[city.team] += len(city.city_cells)

        # if tied, count by units
        team_0 = (
            city_tile_count[0],
            len(self.env.game.get_teams_units(0)),
        )
        team_1 = (
            city_tile_count[1],
            len(self.env.game.get_teams_units(1)),
        )
        if (team_0 == (0, 0)) or (team_1 == (0, 0)):
            return False

        return True

    def __str__(self):
        print(self.env.current_step)
        return self.env.game.map.get_map_string()

    def step(self, actions, from_action_code: bool = True):

        # state transition
        self.update(game=self.env.game, last_actions=actions)
        self.env.current_step += 1
        map_size = (self.env.game.map.height, self.env.game.map.width)
        current_action_plan = np.ones(map_size, dtype=bool)

        if from_action_code:
            action_codes = copy.deepcopy(actions)
            actions = {0: [], 1: []}
            for team in range(self.NUM_AGENTS):
                unit_order = self.unit_order[team]
                our_city = crop_state(
                    self.obs[team]["image"][8] > 0, game=self.env.game
                )

                # check the observation is the lastest
                assert unit_order.step == self.env.current_step - 1
                for seq_ind, unit_id in enumerate(unit_order.order_strings):
                    if seq_ind >= self.unit_length:
                        break
                    unit = self.env.game.state["teamStates"][team]["units"][unit_id]
                    action_code = action_codes[team][seq_ind]
                    if isinstance(action_code, torch.Tensor):
                        action_code = action_code.numpy()

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
                                game=self.env.game,
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
                                game=self.env.game,
                                unit=unit,
                                city_tile=None,
                                team=team,
                            )
                        )

        # Get the next observation
        is_game_over = False
        is_game_error = False
        # action buffering
        for agent in self.env.match_controller.agents:
            game = self.env.game
            team = agent.team
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
                    b_active=self.obs[team]["image"], game=game
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
            self.env.match_controller.take_actions(actions[agent.team] + city_actions)
            (_, _, _, is_new_turn) = next(self.env.match_generator)
            assert not is_new_turn

            self.hist_folder[team].update(
                current_state=self.obs[team]["image"], env_step=game.state["turn"]
            )

        try:
            # actual env step
            (_, _, _, is_new_turn) = next(self.env.match_generator)
            assert is_new_turn
            # is_game_over = not self.check_is_continue()

        except StopIteration:
            # The game episode is done.
            is_game_over = True
        except GameStepFailedException:
            # Game step failed, assign a game lost reward to not incentivise this
            is_game_over = True
            is_game_error = True

        self.env.is_game_over = is_game_over or is_game_error

        return is_game_error

    def turns(self):
        # players to move
        # return [p for p in self.players() if self.obs_list[-1][p]["status"] == "ACTIVE"]
        return [p for p in self.players()]

    def terminal(self):
        return self.env.is_game_over

    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        reward = {}
        for team in self.players():
            reward[team] = self.reward_holder[team](game=self.env.game)

        return reward

    def outcome(self):
        # return terminal outcomes
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        win_agent, agent_city_tile_count = calc_match_result(env=self.env)
        # assert win_agent == self.env.game.get_winning_team()
        if (win_agent != self.env.game.get_winning_team()) and (
            agent_city_tile_count != [0, 0]
        ):
            print("here")

        # rewards = {o["observation"]["index"]: o["reward"] for o in self.obs_list[-1]}
        # outcomes = {p: 0 for p in self.players()}
        # for p, r in rewards.items():
        #     for pp, rr in rewards.items():
        #         if p != pp:
        #             if r > rr:
        #                 outcomes[p] += 1 / (self.NUM_AGENTS - 1)
        #             elif r < rr:
        #                 outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        self.win_agent_hist[win_agent] += 1

        outcomes = {0: 0, 1: 0}
        outcomes[self.env.game.get_winning_team()] += 1
        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def net(self, is_debug: bool = False):
        # context initialization
        with initialize(config_path="./src/config"):
            conf = compose(config_name="config")
            print(OmegaConf.to_yaml(conf))

        lit_model = LitModel(conf=conf)
        if is_debug:
            raise NotImplementedError
        else:
            model_class = partial(
                PolicyValueNet,
                encoder=lit_model.model.encoder,
                decoder=lit_model.model.decoder,
            )
            return model_class

    def observation(self, player, unit: Unit = None):
        # if player is None:
        #     player = 0

        is_new_turn = True
        game = self.env.game
        team = player
        self.is_xy_order = False
        skip_active_unit_drawing = False if unit is not None else True

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
        if not skip_active_unit_drawing:
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
            input_sequence = None
            orig_length = None
            sequence_mask = None
            unit_order = [unit.id]
            self.obs[player] = {
                "image": b_active.astype(np.float32),
                "unit_id": unit.id,
            }
            return self.obs[player]

        else:
            can_act_units = get_can_act_units(game=game, team=team)
            if len(can_act_units) == 0:
                self.obs[player] = {
                    "image": b_active.astype(np.float32),
                    "input_sequence": np.zeros((self.unit_length, 4), dtype=np.float32),
                    # "orig_length": 0,
                    "sequence_mask": np.zeros((self.unit_length, 1), dtype=np.int64),
                    "rule_mask": np.zeros(
                        (self.unit_length, self.action_length()), dtype=np.int64
                    ),
                }
                self.unit_order[player] = UnitOrder(
                    step=self.env.current_step, order_strings=[]
                )
                if self.use_hist:
                    hist_right, hist_left = self.hist_folder[team].get_hist()
                    self.obs[player]["image"][:2] = hist_left
                    self.obs[player]["image"] = np.concatenate(
                        [self.obs[player]["image"], hist_right], axis=0
                    )
                return self.obs[player]

            ordered_unit_points, ordered_units = order_unit(
                units=list(can_act_units.values()),
                is_debug=False,
                random_start=False,
            )

            (
                orig_length,
                sequence_mask,
                input_sequence,
                _,
                unit_order,
                action_masks,
            ) = generate_sequence(
                game=game,
                state=b_active,
                can_act_units=can_act_units,
                ordered_units=ordered_units,
                max_sequence=self.unit_length,  # or len(can_act_units)
                input_size=tuple(self.observation_space.shape[1:]),
                # input_size=self.input_size,
                # no_action=self.no_action,
                # in_features=self.in_features,
                # ignore_class_index=self.ignore_class_index,
                actions=None,
                action_length=self.action_length(),
            )

            self.obs[player] = {
                "image": b_active.astype(np.float32),
                "input_sequence": input_sequence.astype(np.float32),
                # "orig_length": orig_length,
                "sequence_mask": sequence_mask.astype(np.int64),
                "rule_mask": action_masks.astype(np.int64),
            }
            self.unit_order[player] = UnitOrder(
                step=self.env.current_step, order_strings=unit_order
            )
            if self.use_hist:
                hist_right, hist_left = self.hist_folder[team].get_hist()
                self.obs[player]["image"][:2] = hist_left
                self.obs[player]["image"] = np.concatenate(
                    [self.obs[player]["image"], hist_right], axis=0
                )

            return self.obs[player]

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


if __name__ == "__main__":
    e = Environment()

    for _ in range(100):
        e.reset()
        e.use_hist = False
        model, opponent = e.net(is_debug=True)
        model = model.eval()
        opponent.model = opponent.model.eval()
        while not e.terminal():
            print(e)
            # resource
            # plt.imshow(obs_0["image"][12:15].transpose(1, 2, 0))
            # unit
            # plt.imshow(obs_0["image"][2:5].transpose(1, 2, 0))
            actions = {0: [], 1: []}

            team = e.env.match_controller.agents[0].team
            obs_0 = e.observation(player=team)
            with torch.no_grad():
                out0 = model(
                    torch.Tensor(obs_0["image"]).unsqueeze(0),
                    aux_inputs=torch.Tensor(obs_0["input_sequence"]).unsqueeze(0),
                )
            action_logit = out0["outputs"].squeeze().numpy()
            # action_logit = (
            #     action_logit
            #     - (obs_0["rule_mask"] == 0)[:, : action_logit.shape[1]] * 1e32
            # )
            for seq_ind, unit_id in enumerate(e.unit_order[team].order_strings):
                if seq_ind >= e.unit_length:
                    break
                unit = e.env.game.state["teamStates"][team]["units"][unit_id]
                action_code = np.argmax(action_logit[seq_ind])
                actions[team].append(action_code)
                # actions[team].append(
                #     e.action_code_to_action(
                #         action_code.numpy(),
                #         game=e.env.game,
                #         unit=unit,
                #         city_tile=None,
                #         team=team,
                #     )
                # )

            team = e.env.match_controller.agents[1].team
            obs_0 = e.observation(player=team)
            with torch.no_grad():
                out0 = model(
                    torch.Tensor(obs_0["image"]).unsqueeze(0),
                    aux_inputs=torch.Tensor(obs_0["input_sequence"]).unsqueeze(0),
                )
            action_logit = out0["outputs"].squeeze().numpy()
            # action_logit = (
            #     action_logit
            #     - (obs_0["rule_mask"] == 0)[:, : action_logit.shape[1]] * 1e32
            # )
            for seq_ind, unit_id in enumerate(e.unit_order[team].order_strings):
                if seq_ind >= e.unit_length:
                    break
                unit = e.env.game.state["teamStates"][team]["units"][unit_id]
                action_code = np.argmax(action_logit[seq_ind])
                actions[team].append(action_code)
                # actions[team].append(
                #     e.action_code_to_action(
                #         action_code.numpy(),
                #         game=e.env.game,
                #         unit=unit,
                #         city_tile=None,
                #         team=team,
                #     )
                # )

            # team = e.env.match_controller.agents[1].team
            # can_act_units = get_can_act_units(game=e.env.game, team=team)
            # for unit_id, unit in can_act_units.items():
            #     obs_1 = e.observation(player=team, unit=unit)
            #     img = torch.Tensor(obs_1["image"]).transpose(2, 1).unsqueeze(0)
            #     with torch.no_grad():
            #         out1 = opponent.model(img)
            #     action_code = out1.squeeze().max(dim=-1)[1]
            #     # actions[team].append(action_code)
            #     actions[team].append(
            #         e.action_code_to_action(
            #             action_code.numpy(),
            #             game=e.env.game,
            #             unit=unit,
            #             city_tile=None,
            #             team=team,
            #         )
            #     )
            # actions = {p: e.legal_actions(p) for p in e.turns()}

            # print([[e.action2str(a, p) for a in alist] for p, alist in actions.items()])
            # e.step({p: random.choice(alist) for p, alist in actions.items()})
            e.step(actions, from_action_code=True)
            e.reward()
        print(e)
        print(e.outcome())

    print(e.win_agent_hist)
