import time
import traceback
from typing import Dict, List, NamedTuple, Optional

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


class UnitOrder(NamedTuple):
    step: int
    order_strings: List[str]


class Player:
    def __init__(self, research_points: int, city_tile_count: int):
        self.research_points = research_points
        self.city_tile_count = city_tile_count

    def researched_uranium(self):
        return (
            self.research_points
            >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
        )


def get_can_act_units(game: Game, team: int = 0) -> Dict[str, Unit]:
    units = game.state["teamStates"][team]["units"]
    can_act_units = {unit_id: unit for unit_id, unit in units.items() if unit.can_act()}
    return can_act_units


class Reward:
    def __init__(self, team: int = 0):
        """
        from
        https://github.com/glmcdona/LuxPythonEnvGym/blob/main/examples/agent_policy.py
        """
        self.team = team
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.is_last_turn = False

    def __call__(self, game: Game):
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1

        rewards = {}

        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.025
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * 0.1
        self.city_tiles_last = city_tile_count

        # # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = (
            fuel_collected - self.fuel_collected_last
        ) / 20000
        self.fuel_collected_last = fuel_collected

        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0

        # if is_game_finished:
        #     self.is_last_turn = True
        #     rewards["rew/r_city_tiles_end"] = city_tile_count

        #     """
        #     # Example of a game win/loss reward instead
        #     if game.get_winning_team() == self.team:
        #         rewards["rew/r_game_win"] = 100.0 # Win
        #     else:
        #         rewards["rew/r_game_win"] = -100.0 # Loss
        #     """

        reward = 0
        for name, value in rewards.items():
            reward += value  # {player_id: reward}
        return reward * 0.5


class MatchControllerHandyrl(MatchController):
    def __init__(self, game, agents=[None, None], replay_validate=None) -> None:
        """

        :param game:
        :param agents:
        """
        super().__init__(game, agents=agents, replay_validate=replay_validate)

    def run_to_next_observation(self):
        """
        override original method for handyrl env
        """
        game_over = False
        is_first_turn = True
        while not game_over:
            turn = self.game.state["turn"]

            # Run pre-turn agent events to allow for them to handle running the turn instead (used in a kaggle submission agent)
            for agent in self.agents:
                agent.pre_turn(self.game, is_first_turn)

            # Process any pending action sequences to automatically apply actions to units for this turn
            for id in list(self.action_sequences.keys()):
                raise ValueError("action_sequences is not valid")

            # Run agent.turn_heurstics() to apply any agent heristics to give units orders
            for agent in self.agents:
                agent.turn_heurstics(self.game, is_first_turn)

            # Process this turn
            new_turn = True
            yield None, None, None, new_turn
            for agent in self.agents:
                if agent.get_agent_type() == Constants.AGENT_TYPE.AGENT:
                    new_turn = False
                    # Call the agent for the set of actions
                    # actions = agent.process_turn(self.game, agent.team)
                    # self.take_actions(actions)
                    yield None, None, agent.team, new_turn

                elif agent.get_agent_type() == Constants.AGENT_TYPE.LEARNING:
                    # Yield the game to make a decision, since the learning environment is the function caller
                    new_turn = True
                    start_time = time.time()

                    units = self.game.state["teamStates"][agent.team]["units"].values()
                    for unit in units:
                        if unit.can_act():
                            # RL training agent that is controlling the simulation
                            # The enviornment then handles this unit, and calls take_action() to buffer a requested action
                            yield unit, None, unit.team, new_turn
                            new_turn = False

                    cities = self.game.cities.values()
                    for city in cities:
                        if city.team == agent.team:
                            for cell in city.city_cells:
                                city_tile = cell.city_tile
                                if city_tile.can_act():
                                    # RL training agent that is controlling the simulation
                                    # The enviornment then handles this city, and calls take_action() to buffer a requested action
                                    yield None, city_tile, city_tile.team, new_turn
                                    new_turn = False

                    time_taken = time.time() - start_time

            # Reset the can_act overrides for all units and city_tiles
            units = list(self.game.state["teamStates"][0]["units"].values()) + list(
                self.game.state["teamStates"][1]["units"].values()
            )
            for unit in units:
                unit.set_can_act_override(None)
            for city in self.game.cities.values():
                for cell in city.city_cells:
                    city_tile = cell.city_tile.set_can_act_override(None)

            is_first_turn = False

            # Now let the game actually process the requested actions and play the turn
            try:
                # Run post-turn agent events to allow for them to handle running the turn instead (used in a kaggle submission agent)
                self.accumulated_stats = dict(
                    {Constants.TEAM.A: {}, Constants.TEAM.B: {}}
                )
                handled = False
                for agent in self.agents:
                    if agent.post_turn(self.game, self.action_buffer):
                        handled = True

                if not handled:
                    game_over = self.game.run_turn_with_actions(self.action_buffer)
            except Exception as e:
                # Log exception
                self.log_error("ERROR: Critical error occurred in turn simulation.")
                self.log_error(repr(e))
                self.log_error(
                    "".join(traceback.format_exception(None, e, e.__traceback__))
                )
                raise GameStepFailedException(
                    "Critical error occurred in turn simulation."
                )

            self.action_buffer = []

            if self.replay_validate is not None:
                self.game.process_updates(
                    self.replay_validate["steps"][turn + 1][0]["observation"][
                        "updates"
                    ],
                    assign=False,
                )


def default_city_action(
    game: Game, team: int = 0, player_tiles: Optional[list] = None
) -> list:
    actions = []

    # Inference the model per-unit
    units = game.state["teamStates"][team]["units"].values()
    unit_count = len(units)

    # Inference the model per-city
    cities = game.cities.values()
    player_cities = [city for city in cities if city.team == team]
    player = Player(
        research_points=game.state["teamStates"][team]["researchPoints"],
        city_tile_count=sum([len(city.city_cells) for city in player_cities]),
    )
    acted_cities = []
    if player_tiles is not None:
        for city_tile in player_tiles:
            assert city_tile.can_act()
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
            acted_cities.append(city_tile)
        assert unit_count <= player.city_tile_count

    for city in player_cities:
        for cell in city.city_cells:
            city_tile = cell.city_tile
            if city_tile.can_act() and (city_tile not in acted_cities):
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
    return actions
