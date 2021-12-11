import copy
import json
import pickle
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import torch
from luxai2021.game.city import City
from luxai2021.game.constants import Constants, LuxMatchConfigs_Default
from luxai2021.game.game import Game
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position
from luxai2021.game.unit import Unit
from omegaconf import OmegaConf

from src.dataset.utils.process_json import DIRECTION_LABELS, label_to_action, to_label
from src.rl.utils import generate_sequence, order_unit

INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
DIRECTIONS = Constants.DIRECTIONS

CH_MAP = {
    "ActiveUnitPos": 0,
    "ActiveUnitCapa": 1,
    "UnitPos": 2,
    "UnitCooldown": 3,
    "UnitCapa": 4,
    "UnitPos_opp": 5,
    "UnitCooldown_opp": 6,
    "UnitCapa_opp": 7,
    "CityPos": 8,
    "CityLightUpTurns": 9,
    "CityPos_opp": 10,
    "CityLightUpTurns_opp": 11,
    Constants.RESOURCE_TYPES.WOOD: 12,
    Constants.RESOURCE_TYPES.COAL: 13,
    Constants.RESOURCE_TYPES.URANIUM: 14,
    Constants.INPUT_CONSTANTS.RESEARCH_POINTS: 15,  # player 15, oppnent 16
    Constants.INPUT_CONSTANTS.RESEARCH_POINTS + "_opp": 16,  # player 15, oppnent 16
    "TurnsInDay": 17,
    "Turns": 18,
    "InputRange": 19,
}

MONO_MAP = [
    Constants.INPUT_CONSTANTS.RESEARCH_POINTS,
    Constants.INPUT_CONSTANTS.RESEARCH_POINTS + "_opp",
    "TurnsInDay",
    "Turns",
    # "InputRange",
]


class GameObs(NamedTuple):
    game: Game
    player_index: int
    b: np.ndarray
    unit_stack_map: np.ndarray


class LuxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        obses: dict,
        game_obses: Dict[str, Union[GameObs, Path]],
        samples: np.ndarray,
        num_state_features: int = 20,
        input_size: Tuple[int, int] = (32, 32),
        is_xy_order: bool = False,
        cache_obs_in_memory: bool = False,
        transforms: Optional[albu.Compose] = None,
        random_crop: bool = False,
        action_hists: Optional[Dict[str, List[List[str]]]] = None,
        ignore_class_index: int = 20,
        max_sequence: int = 128,
        decoder_in_features: int = 4,
        skip_active_unit_drawing: bool = False,
        no_action_index: int = 5,
        random_start_ordering: bool = False,
        no_action_droprate: float = 0.0,
    ):

        """
        modify from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
        per unit (state, action) pair
        state, (20, 32, 32) for all posible maps, (8, 8), (16, 16) with zero padding
        first channel of the state is the target unit position
        """
        self.obses = obses
        self.game_obses = game_obses
        self.samples = samples
        self.input_size = input_size
        self.feat_map_generator = partial(
            make_input_from_obs,
            num_state_features=num_state_features,
            input_size=input_size,
            is_xy_order=is_xy_order,
            skip_active_unit_drawing=skip_active_unit_drawing,
        )
        self.cache_obs_in_memory = cache_obs_in_memory
        self.transforms = transforms
        self.random_crop = random_crop
        self.action_hists = action_hists
        self.in_features = decoder_in_features
        self.ignore_class_index = ignore_class_index
        self.max_sequence = max_sequence
        self.skip_active_unit_drawing = skip_active_unit_drawing
        self.no_action = no_action_index
        self.random_start_ordering = random_start_ordering
        self.undefined_action = no_action_index + 1
        self.no_action_droprate = no_action_droprate

    def __len__(self):
        return len(self.samples)

    def _get_position_from_mask(self, mask):
        pos = np.where(mask == 1.0)
        pos = Position(x=pos[1][0], y=pos[0][0])
        return pos

    @staticmethod
    def random_map_crop(state: np.ndarray, game: Game) -> np.ndarray:
        input_size = state.shape[1:]

        if (game.map.height == input_size[0]) & (game.map.width == input_size[1]):
            return state

        y_shift_top, x_shift_left = calc_game_map_shift(
            input_size=input_size, game=game
        )

        y_shift_bottom = input_size[0] - game.map.height - y_shift_top
        x_shift_right = input_size[1] - game.map.width - x_shift_left
        # random.random [0, 1), so need +1 for int
        y_crop_end = input_size[0] - int(random.random() * (y_shift_bottom + 1))
        x_crop_end = input_size[1] - int(random.random() * (x_shift_right + 1))

        crop_state = state[
            :,
            int(random.random() * (y_shift_top + 1)) : y_crop_end,
            int(random.random() * (x_shift_left + 1)) : x_crop_end,
        ]
        assert (
            crop_state[CH_MAP["InputRange"], :].sum()
            == state[CH_MAP["InputRange"], :].sum()
        )
        return crop_state

    @staticmethod
    def pad_image(
        image: np.ndarray,
        input_size: List[int],
        constant_values: float = 255.0,
        channel_last: bool = False,
        **kwargs,
    ):
        if channel_last:
            pad_size = (input_size[0] - image.shape[0], input_size[1] - image.shape[1])
        else:
            pad_size = (input_size[0] - image.shape[1], input_size[1] - image.shape[2])

        # def _update_pad_size(pad_size: Tuple[int, int]):
        #     if channel_last:
        #         pad_size = [[0, pad_size[0]], [0, pad_size[1]], [0, 0]]
        #     else:
        #         pad_size = [[0, 0], [0, pad_size[0]], [0, pad_size[1]]]
        #     return pad_size

        if np.any(np.array(pad_size) > 0):
            if channel_last:
                image = np.pad(
                    image,
                    [[0, pad_size[0]], [0, pad_size[1]], [0, 0]],
                    constant_values=constant_values,
                    # mode="reflect",
                )
            else:
                image = np.pad(
                    image,
                    [[0, 0], [0, pad_size[0]], [0, pad_size[1]]],
                    constant_values=constant_values,
                    # mode="reflect",
                )

            # image[:, :, orig_width:] = constant_values
        return image

    @staticmethod
    def load_orig_json(
        episode_id: int,
        episode_dir: str = "../input/lux-ai-episodes/",
    ):
        episode = Path(episode_dir, str(episode_id) + ".json")
        with episode.open() as f:
            episode = json.load(f)
        return episode

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        # obs = self.obses[obs_id]
        # state = self.feat_map_generator(obs=obs, unit_id=unit_id)
        episode_id, step = (int(id) for id in obs_id.split("_"))
        team_id = self.obses[obs_id]["player"]

        # episode = LuxDataset.load_orig_json(episode_id=episode_id)

        # get game object
        game_obs = self.game_obses[obs_id]
        if isinstance(game_obs, Path):
            with game_obs.open(mode="rb") as f:
                game_obs = pickle.load(file=f)
            if self.cache_obs_in_memory:
                self.game_obses[obs_id] = game_obs

        state = make_input_from_obs(
            obs=None,
            unit_id=unit_id,
            pre_computed_game_obs=game_obs,
            skip_active_unit_drawing=self.skip_active_unit_drawing,
        )

        actions, can_act_units = self.get_actions_in_this_turn(
            episode_id, step, team_id, game_obs, undefined_label=self.undefined_action
        )
        if self.no_action_droprate > 0.0:
            can_act_units = {
                unit_id: unit
                for unit_id, unit in can_act_units.items()
                if (unit_id in list(actions.keys()))
                or (random.random() > self.no_action_droprate)
            }

        ordered_unit_points, ordered_units = order_unit(
            units=list(can_act_units.values()),
            is_debug=False,
            random_start=self.random_start_ordering,
        )
        (
            orig_length,
            sequence_mask,
            input_sequence,
            output_sequence,
            _,
            action_masks,
        ) = generate_sequence(
            game=game_obs.game,
            state=state,
            can_act_units=can_act_units,
            ordered_units=ordered_units,
            max_sequence=self.max_sequence,  # or len(can_act_units)
            input_size=self.input_size,
            no_action=self.no_action,
            in_features=self.in_features,
            ignore_class_index=self.ignore_class_index,
            actions=actions,
        )

        if self.random_crop:
            state = LuxDataset.random_map_crop(state=state, game=game_obs.game)
            state = LuxDataset.pad_image(
                image=state, input_size=self.input_size, constant_values=0.0
            )
            # for mono ch, we fix padding value with original one
            mono_ch = [CH_MAP[map_key] for map_key in MONO_MAP]
            state[mono_ch] = np.max(state[mono_ch], axis=(1, 2), keepdims=True)

        # game_obs.game.state["teamStates"][game_obs.player_index]["units"][unit_id].pos.x
        # game_obs.game.state["teamStates"][game_obs.player_index]["units"][unit_id].pos.y
        if self.transforms is not None:
            cur_mask = state[CH_MAP["ActiveUnitPos"]]
            cur_pos = self._get_position_from_mask(mask=state[CH_MAP["ActiveUnitPos"]])
            mask = np.zeros(state.shape[1:], dtype=np.float32)

            if action != 4:
                new_pos = cur_pos.translate(
                    direction=label_to_action(label=action), units=1
                )
            else:
                new_pos = cur_pos

            mask[new_pos.y, new_pos.x] = 1.0
            mask_ = cur_mask + mask * 0.5  # for dubug

            augmented = self.transforms(image=state.transpose(1, 2, 0), mask=mask)
            state = augmented["image"].transpose(2, 0, 1)
            mask = augmented["mask"]

            cur_mask = state[CH_MAP["ActiveUnitPos"]]
            aug_new_pos = self._get_position_from_mask(mask=mask)
            aug_pos = self._get_position_from_mask(mask=state[CH_MAP["ActiveUnitPos"]])
            mask_ = cur_mask + mask * 0.5  # for dubug
            # build city is no change
            if action != 4:
                direction = aug_pos.direction_to(target_pos=aug_new_pos)
                action = DIRECTION_LABELS[direction]

        state = torch.tensor(state, dtype=torch.float32)
        target = torch.tensor(action, dtype=torch.int64)
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        sequence_mask = torch.tensor(sequence_mask, dtype=torch.float32)
        output_sequence = torch.tensor(output_sequence, dtype=torch.int64)
        # output_sequence = torch.nn.functional.one_hot(
        #     output_sequence, num_classes=self.pad_class + 1
        # )

        return {
            "id": obs_id,
            "image": state,
            "target": target,
            "unit_id": unit_id,
            "input_sequence": input_sequence,
            "output_sequence": output_sequence,
            "orig_length": orig_length,
            "sequence_mask": sequence_mask,
        }

    def get_actions_in_this_turn(
        self, episode_id, step, team_id, game_obs, undefined_label: int = 5
    ):
        # get all actions in this turn
        actions = {}
        for act in self.action_hists.get(episode_id)[step][team_id]:
            action_type = act.split(" ")[0]
            if action_type not in [
                Constants.ACTIONS.MOVE,
                Constants.ACTIONS.BUILD_CITY,
            ]:
                continue

            unit_id, label = to_label(act)
            if label is None:
                label = undefined_label
            actions[unit_id] = label

        units = game_obs.game.state["teamStates"][team_id]["units"]
        can_act_units = {
            unit_id: unit for unit_id, unit in units.items() if unit.can_act()
        }

        return actions, can_act_units


# Input for Neural Network
def make_input(
    obs: dict,
    unit_id: str,
    num_state_features: int = 20,
    input_size: Tuple[int, int] = (32, 32),
):
    """
    from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    """
    # x, y coordinate definition... x-0, y-1 or y-0, x-1
    # keep it as original, so no modification for xy order
    width, height = obs["width"], obs["height"]
    x_shift = (input_size[1] - width) // 2
    y_shift = (input_size[0] - height) // 2
    cities = {}
    b = np.zeros((num_state_features, input_size[0], input_size[1]), dtype=np.float32)

    for update in obs["updates"]:
        strs = update.split(" ")
        input_identifier = strs[0]

        if input_identifier == "u":
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (1, (wood + coal + uranium) / 100)
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs["player"]) % 2 * 3
                b[idx : idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100,
                )
        elif input_identifier == "ct":
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs["player"]) % 2 * 2
            b[idx : idx + 2, x, y] = (1, cities[city_id])
        elif input_identifier == "r":
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{"wood": 12, "coal": 13, "uranium": 14}[r_type], x, y] = amt / 800
        elif input_identifier == "rp":
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs["player"]) % 2, :] = min(rp, 200) / 200
        elif input_identifier == "c":
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

    # Day/Night Cycle
    b[17, :] = obs["step"] % 40 / 40
    # Turns
    b[18, :] = obs["step"] / 360
    # Map Size

    b[19, x_shift : 32 - x_shift, y_shift : 32 - y_shift] = 1
    state = make_input_from_obs(obs=obs, unit_id=unit_id, is_xy_order=True)
    assert np.all(state == b)
    return b


class GameMiddleState(Game):
    def __init__(self, configs=LuxMatchConfigs_Default):
        # generate empty map with minimal size
        # Use an empty map, because the updates will fill the map out
        MIN_SIZE = (12, 12)
        configs["width"] = int(MIN_SIZE[1])
        configs["height"] = int(MIN_SIZE[0])
        configs["mapType"] = Constants.MAP_TYPES.EMPTY

        super().__init__(configs=configs)

    def process_updates(self, updates, assign=True):

        if updates is None:
            return

        # Loop through updating the game from the list of updates
        # Implements /kits/python/simple/lux/game.py -> _update()
        for update in updates:
            if update == "D_DONE":
                break
            strings = update.split(" ")

            input_identifier = strings[0]
            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strings[1])
                research_points = int(strings[2])
                if assign:
                    self.state["teamStates"][team]["researchPoints"] = research_points
                else:
                    assert (
                        self.state["teamStates"][team]["researchPoints"]
                        == research_points
                    )

                if (
                    int(strings[2])
                    >= self.configs["parameters"]["RESEARCH_REQUIREMENTS"]["COAL"]
                ):
                    if assign:
                        self.state["teamStates"][team]["researched"]["coal"] = True
                    else:
                        assert (
                            self.state["teamStates"][team]["researched"]["coal"] == True
                        )

                if (
                    int(strings[2])
                    >= self.configs["parameters"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
                ):
                    if assign:
                        self.state["teamStates"][team]["researched"]["uranium"] = True
                    else:
                        assert (
                            self.state["teamStates"][team]["researched"]["uranium"]
                            == True
                        )

            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strings[1]
                x = int(strings[2])
                y = int(strings[3])
                amt = int(float(strings[4]))
                if assign:
                    self.map.add_resource(x, y, r_type, amt)
                else:
                    cell = self.map.get_cell(x, y)
                    assert cell.resource.amount == amt
                    assert cell.resource.type == r_type

            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unit_type = int(strings[1])
                team = int(strings[2])
                unit_id = strings[3]
                x = int(strings[4])
                y = int(strings[5])
                cooldown = float(strings[6])
                wood = int(strings[7])
                coal = int(strings[8])
                uranium = int(strings[9])
                if assign:
                    if unit_type == Constants.UNIT_TYPES.WORKER:
                        self.spawn_worker(
                            team,
                            x,
                            y,
                            unit_id,
                            cooldown=cooldown,
                            cargo={"wood": wood, "uranium": uranium, "coal": coal},
                        )
                    elif unit_type == Constants.UNIT_TYPES.CART:
                        self.spawn_cart(
                            team,
                            x,
                            y,
                            unit_id,
                            cooldown=cooldown,
                            cargo={"wood": wood, "uranium": uranium, "coal": coal},
                        )
                else:
                    cell = self.map.get_cell(x, y)
                    assert len(cell.units) > 0
                    assert unit_id in [
                        u.id for u in cell.units.values()
                    ], f"unit id {unit_id} missplaced"

            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strings[1])
                city_id = strings[2]
                fuel = float(strings[3])
                light_upkeep = float(strings[4])  # Unused
                if assign:
                    self.cities[city_id] = City(team, self.configs, None, city_id, fuel)
                else:
                    assert city_id in self.cities

            elif input_identifier == INPUT_CONSTANTS.CITY_TILES:
                team = int(strings[1])
                city_id = strings[2]
                x = int(strings[3])
                y = int(strings[4])
                cooldown = float(strings[5])
                city = self.cities[city_id]
                cell = self.map.get_cell(x, y)
                if assign:
                    self.spawn_city_tile_from_middle(
                        team,
                        x,
                        y,
                        self.cities[city_id],
                        city_id=city_id,
                        cooldown=cooldown,
                    )
                    # cell.set_city_tile(team, city_id, cooldown)
                    # city.add_city_tile(cell)
                    self.stats["teamStats"][team]["cityTilesBuilt"] += 1
                else:
                    assert cell.city_tile.city_id == city_id
                    assert cell in city.city_cells

            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strings[1])
                y = int(strings[2])
                road = float(strings[3])
                cell = self.map.get_cell(x, y)
                if cell not in self.cells_with_roads:
                    self.cells_with_roads.add(cell)
                if assign:
                    cell.road = road
                else:
                    assert cell.get_road() == road

    def spawn_city_tile_from_middle(
        self, team, x, y, city: City, city_id: str = None, cooldown: int = 0
    ):
        """
        Spawns new city tile
        Implements src/Game/index.ts -> Game.spawnCityTile()
        """
        cell = self.map.get_cell(x, y)

        # now update the cities field accordingly
        adj_cells = self.map.get_adjacent_cells(cell)

        city_ids_found = []

        adj_same_team_city_tiles = []
        for cell2 in adj_cells:
            if cell2.is_city_tile() and cell2.city_tile.team == team:
                adj_same_team_city_tiles.append(cell2)
                if cell2.city_tile.city_id not in city_ids_found:
                    city_ids_found.append(cell2.city_tile.city_id)

        # if no adjacent city cells of same team, generate new city
        if len(adj_same_team_city_tiles) == 0:
            # for update from middle state, city tiles  in city-culster are
            # itrated rondomly so we could not expect adj_same_team_citye_tiles > 0
            # within the same culster
            # city = City(team, self.configs, self.global_city_id_count + 1, fuel=fuel)

            if city_id is not None:
                city.id = city_id
            else:
                self.global_city_id_count += 1

            cell.set_city_tile(team, city_id, cooldown)
            city.add_city_tile(cell)
            # self.cities[city.id] = city
            return cell.city_tile

        else:
            # otherwise add tile to city
            city_id = adj_same_team_city_tiles[0].city_tile.city_id
            city = self.cities[city_id]
            cell.set_city_tile(team, city_id, cooldown)

            # update adjacency counts for bonuses
            cell.city_tile.adjacent_city_tiles = len(adj_same_team_city_tiles)
            for adjCell in adj_same_team_city_tiles:
                adjCell.city_tile.adjacent_city_tiles += 1
            city.add_city_tile(cell)

            # update all merged cities' cells with merged city_id, move to merged city and delete old city
            for local_id in city_ids_found:
                if local_id != city_id:
                    old_city = self.cities[local_id]
                    for cell3 in old_city.city_cells:
                        cell3.city_tile.city_id = city_id
                        city.add_city_tile(cell3)

                    city.fuel += old_city.fuel
                    self.cities.pop(old_city.id)

            return cell.city_tile


def generate_game_state_from_obs(obs: Dict[str, Any], configs=LuxMatchConfigs_Default):
    """
    from
    https://github.com/glmcdona/LuxPythonEnvGym/blob/main/luxai2021/env/agent.py#L223
    """
    game = GameMiddleState(configs=configs)

    # overides map def
    game.configs["width"] = int(obs["width"])
    game.configs["height"] = int(obs["height"])
    # Use an empty map, because the updates will fill the map out
    game.configs["mapType"] = Constants.MAP_TYPES.EMPTY

    # Reset the game to the specified state. Don't increment turn counter on first turn of game.
    game.reset(updates=obs["updates"], increment_turn=False)
    game.state["turn"] = obs["step"]
    game.global_unit_id_count = obs.get("globalUnitIDCount", 0)
    game.global_city_id_count = obs.get("globalCityIDCount", 0)

    return game


def make_input_from_obs(
    obs: dict,
    unit_id: str,
    num_state_features: int = 20,
    input_size: Tuple[int, int] = (32, 32),
    is_xy_order: bool = False,
    pre_computed_game_obs: Optional[Dict[str, GameObs]] = None,
    skip_active_unit_drawing: bool = False,
) -> np.ndarray:
    """
    from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    """

    if pre_computed_game_obs is None:
        game_obs = generate_game_obs_from_obs(
            obs=obs, num_state_features=num_state_features, input_size=input_size
        )
    else:
        game_obs = pre_computed_game_obs

    # add active unit
    b_active = game_obs.b.copy()
    if not skip_active_unit_drawing:
        b_active = update_active_unit_on_input(
            b=b_active,
            game=game_obs.game,
            unit_id=unit_id,
            player_index=game_obs.player_index,
            input_size=input_size,
            unit_stack_map=game_obs.unit_stack_map,
        )
    # only for kaggle public notebook compatibility
    if is_xy_order:
        b_active = b_active.transpose((0, 2, 1))

    return b_active


def generate_game_obs_from_obs(
    obs: dict,
    num_state_features: int = 20,
    input_size: Tuple[int, int] = (32, 32),
    cache_dir: Optional[Path] = None,
    obs_ids: Optional[List[str]] = None,
):
    obs_id = obs.pop("obs_id", None)
    game_configs = copy.deepcopy(LuxMatchConfigs_Default)
    game = generate_game_state_from_obs(obs=obs, configs=game_configs)
    player_index = obs["player"]
    b, unit_stack_map = make_input_from_game_state(
        game=game,
        active_unit_id=None,
        player_index=player_index,
        num_state_features=num_state_features,
        input_size=input_size,
    )
    game_obs = GameObs(
        game=game,
        player_index=player_index,
        b=b,
        unit_stack_map=unit_stack_map,
    )
    if cache_dir is not None:
        assert obs_id is not None
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_path = Path(cache_dir, obs_id + ".pickle")
        with cache_path.open(mode="wb") as f:
            pickle.dump(obj=game_obs, file=f)
        return cache_path
    else:
        return game_obs


#     return {
#         "game": game,
#         "player_index": player_index,
#         "b": b,
#         "unit_stack_map": unit_stack_map,
#     }


def is_oppent_obj(team: int = 0, player_index: int = 0) -> str:
    if team != player_index:
        return "_opp"
    else:
        return ""


def make_input_from_game_state(
    game: Game,
    active_unit_id: Optional[str] = None,
    player_index: int = 0,
    num_state_features: int = 20,
    input_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    # x_shift = (input_size[1] - game.map.width) // 2
    # y_shift = (input_size[0] - game.map.height) // 2
    y_shift, x_shift = calc_game_map_shift(input_size=input_size, game=game)
    b = np.zeros((num_state_features, input_size[0], input_size[1]), dtype=np.float32)
    unit_stack_map = [[[] for x in range(input_size[1])] for y in range(input_size[0])]
    _is_oppent_obj = partial(is_oppent_obj, player_index=player_index)

    # unit
    for team, team_state in game.state["teamStates"].items():
        for uid, unit in team_state["units"].items():
            b, unit_stack_map = process_unit(
                b=b,
                unit=unit,
                team=team,
                # active_unit_id=active_unit_id,
                active_unit_id=None,
                x_shift=x_shift,
                y_shift=y_shift,
                _is_oppent_obj=_is_oppent_obj,
                unit_stack_map=unit_stack_map,
            )

        # [0, 1]

    # resarch point
    for r_cell in game.map.resources:
        x = int(r_cell.pos.x) + x_shift
        y = int(r_cell.pos.y) + y_shift
        b[CH_MAP[r_cell.resource.type], y, x] = r_cell.resource.amount / 800
        # [0, 1] for wood

    # cities & city_tiles
    for city_id, city in game.cities.items():
        light_up_turns = min(city.fuel / city.get_light_upkeep(), 10) / 10
        for c_cell in city.city_cells:
            city_tile = c_cell.city_tile
            x = city_tile.pos.x + x_shift
            y = city_tile.pos.y + y_shift
            b[CH_MAP["CityPos" + _is_oppent_obj(city.team)], y, x] = 1
            b[
                CH_MAP["CityLightUpTurns" + _is_oppent_obj(city.team)], y, x
            ] = light_up_turns

    # resarch point
    for team, team_state in game.state["teamStates"].items():
        b[CH_MAP["rp" + _is_oppent_obj(team)], :] = (
            min(team_state["researchPoints"], 200) / 200
        )
        # [0, 1]

    # Day/Night Cycle
    b[CH_MAP["TurnsInDay"], :] = game.state["turn"] % 40 / 40
    # Turns
    b[CH_MAP["Turns"], :] = game.state["turn"] / 360
    # Map Size
    b[
        CH_MAP["InputRange"],
        y_shift : input_size[0] - y_shift,
        x_shift : input_size[1] - x_shift,
    ] = 1

    return b, unit_stack_map


def process_unit(
    b: np.ndarray,
    unit,
    team: str,
    x_shift: int,
    y_shift: int,
    _is_oppent_obj: Callable,
    active_unit_id: Optional[str] = None,
    unit_stack_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x = int(unit.pos.x) + x_shift
    y = int(unit.pos.y) + y_shift
    # .get_cargo_fuel_value(), get_cargo_space_left
    capa = (
        sum(unit.cargo.values())
        / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
    )

    def _unit_operation(
        b: np.ndarray, team: str, unit, x: int, y: int, capa: float, mode: str = "add"
    ) -> np.ndarray:
        if mode == "add":
            b[CH_MAP["UnitPos" + _is_oppent_obj(team)], y, x] = 1.0
            b[CH_MAP["UnitCooldown" + _is_oppent_obj(team)], y, x] = (
                float(unit.cooldown) / 6.0
            )
            b[CH_MAP["UnitCapa" + _is_oppent_obj(team)], y, x] = capa
        elif mode == "delete":
            b[CH_MAP["UnitPos" + _is_oppent_obj(team)], y, x] = 0.0
            b[CH_MAP["UnitCooldown" + _is_oppent_obj(team)], y, x] = 0.0
            b[CH_MAP["UnitCapa" + _is_oppent_obj(team)], y, x] = 0.0
        else:
            raise NotADirectoryError(f"mode: {mode} is not implemented")
        return b

    if active_unit_id is not None:
        assert unit.id == active_unit_id
        # TODO: cooldown is needed?
        b[CH_MAP["ActiveUnitPos"], y, x] = 1.0
        b[CH_MAP["ActiveUnitCapa"], y, x] = capa
        b = _unit_operation(
            b=b, team=team, unit=unit, x=x, y=y, capa=capa, mode="delete"
        )
        return b, unit_stack_map
    else:
        b = _unit_operation(b=b, team=team, unit=unit, x=x, y=y, capa=capa, mode="add")
        if unit_stack_map is not None:
            unit_stack_map[y][x].append(unit.id)
        return b, unit_stack_map


def calc_game_map_shift(input_size: List[int], game: Game) -> Tuple[int, int]:
    x_shift = (input_size[1] - game.map.width) // 2
    y_shift = (input_size[0] - game.map.height) // 2
    return (y_shift, x_shift)


def update_active_unit_on_input(
    b: np.ndarray,
    game: Game,
    unit_id: str,
    player_index: int = 0,
    input_size: Tuple[int, int] = (32, 32),
    unit_stack_map: Optional[np.ndarray] = None,
) -> np.ndarray:

    # x_shift = (input_size[1] - game.map.width) // 2
    # y_shift = (input_size[0] - game.map.height) // 2
    y_shift, x_shift = calc_game_map_shift(input_size=input_size, game=game)
    active_u = game.state["teamStates"][player_index]["units"][unit_id]
    _is_oppent_obj = partial(is_oppent_obj, player_index=player_index)

    b_active, _ = process_unit(
        b=b,
        unit=active_u,
        team=player_index,
        active_unit_id=unit_id,
        x_shift=x_shift,
        y_shift=y_shift,
        _is_oppent_obj=_is_oppent_obj,
    )
    # for stacking units on the same citytile
    if unit_stack_map is not None:
        stack_list = unit_stack_map[active_u.pos.y + y_shift][
            active_u.pos.x + x_shift
        ].copy()
        # active unit has been embed in another channel, so remove it
        stack_list.remove(unit_id)
        if len(stack_list) > 0:
            try:
                append_unit = game.state["teamStates"][player_index]["units"][
                    stack_list[-1]
                ]
                b_active, _ = process_unit(
                    b=b_active,
                    unit=append_unit,
                    team=player_index,
                    active_unit_id=None,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    _is_oppent_obj=_is_oppent_obj,
                    unit_stack_map=None,  # should be None for unnecessary override
                )
            except Exception as e:
                pass
                # print(e)
                # game_state = game.to_state_object()
                # stack_list = unit_stack_map[active_u.pos.y + y_shift][
                #     active_u.pos.x + x_shift
                # ]
                # active_u.team == game.map.get_cell_by_pos(pos=active_u.pos).city_tile.team
                # active_u.team == game.state["teamStates"][(player_index+1)%2]["units"][stack_list[-1]].team

    return b_active
