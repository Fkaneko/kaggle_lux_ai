import copy
import json
import pickle
import random
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import skimage.measure
import skimage.transform
import torch

# n , s, w, e
# (x, y)
ACTION_DIRECTIONS = [(0, -1), (0, +1), (-1, 0), (1, 0)]


class UnitOrder(NamedTuple):
    step: int
    order_strings: List[str]


def get_cargo_fuel_value(unit):
    """
    Returns the fuel-value of all the cargo this unit has.
    """
    wood_rate = 1
    coal_rate = 10
    uranium_rate = 40
    if hasattr(unit, "get_cargo_fuel_value"):
        return unit.get_cargo_fuel_value()
    else:
        return (
            unit.cargo.wood * wood_rate
            + unit.cargo.coal * coal_rate
            + unit.cargo.uranium * uranium_rate
        )


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )
    if isinstance(tensor, np.ndarray):
        return tensor


# def get_can_act_units(game: Any, team: int = 0) -> Dict[str, Any]:
#     units = game.state["teamStates"][team]["units"]
#     can_act_units = {unit_id: unit for unit_id, unit in units.items() if unit.can_act()}
#     return can_act_units


def get_unit_sequence_obs(
    game,
    player: int,
    b_active: np.ndarray,
    can_act_units: Dict[str, Any],
    turn: int = 0,
    unit_length: int = 64,
    action_dim: int = 5,
    input_dim=4,
):
    """
    Implements getting a observation from the current game for this unit or city
    """

    if len(can_act_units) == 0:
        obs = {
            "image": b_active.astype(np.float32),
            "input_sequence": np.zeros((unit_length, input_dim), dtype=np.float32),
            # "orig_length": 0,
            "sequence_mask": np.zeros((unit_length, 1), dtype=np.int64),
            "rule_mask": np.zeros((unit_length, action_dim), dtype=np.int64),
        }
        unit_order = UnitOrder(step=turn, order_strings=[])
        return obs, unit_order

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
        max_sequence=unit_length,  # or len(can_act_units)
        input_size=b_active.shape[1:],
        # input_size=input_size,
        # no_action=no_action,
        # in_features=in_features,
        # ignore_class_index=ignore_class_index,
        actions=None,
        action_length=action_dim,
    )

    obs = {
        "image": b_active.astype(np.float32),
        "input_sequence": input_sequence.astype(np.float32),
        # "orig_length": orig_length,
        "sequence_mask": sequence_mask.astype(np.int64),
        "rule_mask": action_masks.astype(np.int64),
    }
    unit_order = UnitOrder(step=turn, order_strings=unit_order)
    return obs, unit_order


def order_points(points, units, ind):
    """
    from
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    """
    points_new = [
        points.pop(ind)
    ]  # initialize a new list of points with the known first point
    units_new = [
        units.pop(ind)
    ]  # initialize a new list of points with the known first point
    pcurr = points_new[-1]  # initialize the current point (as the known point)
    while len(points) > 0:
        d = np.linalg.norm(
            np.array(points) - np.array(pcurr), axis=1
        )  # distances between pcurr and all other remaining points
        ind = d.argmin()  # index of the closest point
        points_new.append(points.pop(ind))  # append the closest point to points_new
        units_new.append(units.pop(ind))  # append the closest point to points_new
        pcurr = points_new[-1]  # update the current point
    return points_new, units_new


def order_unit(
    units: List[Any],
    is_debug: bool = False,
    random_start: bool = False,
    split_proba: float = 0.1,
) -> List[Tuple[int, int]]:
    """
    from
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    """
    # assemble the x and y coordinates into a list of (x,y) tuples:
    ordered_units = sorted(units, key=lambda x: (x.pos.y, x.pos.x))
    points = [(unit.pos.x, unit.pos.y) for unit in ordered_units]

    # order the points based on the known first point:
    start_ind = 0
    if random_start:
        start_ind = random.choice(range(len(points)))
    points_new, units_new = order_points(
        points.copy(), units=ordered_units.copy(), ind=start_ind
    )

    if random_start:
        points_new = points_new[::-1]
        if random.random() < split_proba:
            split_ind = random.choice(range(len(points)))
            points_new = points_new[split_ind:] + points_new[:split_ind]

    if is_debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        points = np.stack(points, axis=0)
        xn, yn = np.array(points_new).T
        ax[0].plot(points[:, 0], points[:, 1], "-o")  # original (shuffled) points
        ax[1].plot(xn, yn, "-o")  # new (ordered) points
        ax[0].set_title("Original")
        ax[1].set_title("Ordered")
        ax[0].grid()
        ax[1].grid()
        plt.tight_layout()
        plt.show()
    return points_new, units_new


def generate_sequence(
    game: Any,
    state: np.ndarray,
    can_act_units: Dict[str, Any],
    # ordered_unit_points: List[Tuple[int, int]],
    ordered_units: List[Any],
    input_size: Tuple[int, int] = (32, 32),
    max_sequence: int = 128,
    no_action: int = 6,
    ignore_class_index: int = 20,
    in_features: int = 4,
    actions: Optional[Dict[str, int]] = None,
    action_length: int = 5,
):

    # unit feature
    # (N_u, 4), x, y, act_tpye, dirction/None
    if actions is None:
        actions = {}

    unit_feature = []
    unit_order = []
    action_masks = []

    ban_map = generate_pos_ban_map(b_active=state)
    for unit in ordered_units:
        action_mask = np.ones((action_length,), dtype=np.int64)
        unit_id = unit.id
        y_shift, x_shift = calc_game_map_shift(input_size=input_size, game=game)
        unit_feature.append(
            [
                unit.pos.x + x_shift,
                unit.pos.y + y_shift,
                can_act_units[unit_id].get_cargo_space_left(),
                get_cargo_fuel_value(can_act_units[unit_id]),
                actions.pop(unit_id, no_action),
            ]
        )
        assert (
            # state[CH_MAP["UnitPos"], unit_feature[-1][1], unit_feature[-1][0]]
            state[2, unit_feature[-1][1], unit_feature[-1][0]]
            == 1.0
        )
        unit_order.append(unit_id)
        action_mask[4] = int(can_act_units[unit_id].can_build(game.map))
        action_mask[:4] = not_go_pos(
            pos_y=unit_feature[-1][1], pos_x=unit_feature[-1][0], ban_map=ban_map
        )
        action_masks.append(action_mask)

    assert len(actions) == 0
    unit_feature = np.stack(unit_feature, axis=0)
    action_masks = np.stack(action_masks, axis=0)
    unit_feature = unit_feature[:max_sequence]
    action_masks = action_masks[:max_sequence]
    # debug
    # fig, axes = plt.subplots(1, 3)
    # state_map = state[CH_MAP["UnitPos"]] -  (state[CH_MAP["UnitCooldown"]] > 0).astype(int)
    # cell_map = np.zeros_like(state_map)
    # for x, y in unit_feature[:, :2]:
    #     cell_map[y, x] = 1.
    # axes[0].imshow(state_map)
    # axes[1].imshow(cell_map)
    # axes[2].imshow((state_map - cell_map) * 0.5)

    orig_length = unit_feature.shape[0]
    pad_len = max_sequence - orig_length
    if pad_len > 0:
        unit_feature = np.pad(unit_feature, [[0, pad_len], [0, 0]], constant_values=-1)
        action_masks = np.pad(action_masks, [[0, pad_len], [0, 0]], constant_values=-1)
    sequence_mask = np.zeros_like(unit_feature[:, 0:1])
    sequence_mask[:orig_length, :] = 1.0

    input_sequence = unit_feature[:, :in_features] / 100.0
    output_sequence = unit_feature[:, in_features:]
    output_sequence[output_sequence == -1] = ignore_class_index

    return (
        orig_length,
        sequence_mask,
        input_sequence,
        output_sequence,
        unit_order,
        action_masks,
    )


def calc_game_map_shift(input_size: List[int], game: Any) -> Tuple[int, int]:
    x_shift = (input_size[1] - game.map.width) // 2
    y_shift = (input_size[0] - game.map.height) // 2
    return (y_shift, x_shift)


def crop_state(state: np.ndarray, game: Any, input_size=[32, 32]):
    y_shift, x_shift = calc_game_map_shift(input_size=input_size, game=game)

    if state.ndim == 2:
        state = state[
            y_shift : input_size[1] - y_shift, x_shift : input_size[0] - x_shift
        ]
    elif state.ndim == 3:
        state = state[
            :, y_shift : input_size[1] - y_shift, x_shift : input_size[0] - x_shift
        ]
    return state


def pred_with_onnx(model: ort.InferenceSession, obs: Dict[str, np.ndarray]):
    ort_inputs = {}
    for ort_in in model.get_inputs():
        ort_inputs[ort_in.name] = to_numpy(obs[ort_in.name])[
            np.newaxis,
        ]
    out = model.run(
        None,
        ort_inputs,
    )
    action_logit = out[0]
    action_logit = (action_logit - (obs["rule_mask"] == 0) * 1e32).squeeze(0)
    return action_logit


def get_resource_distribution(b_active: np.ndarray, game: Any) -> np.ndarray:
    # resoure_map = b_active[[12, 13, 14]].transpose(1, 2, 0)
    # resoure_map = b_active[[12, 13, 14]]
    input_size = (32, 32)
    y_shift, x_shift = calc_game_map_shift(input_size=input_size, game=game)

    research_point = int(b_active[[15]].max() * 200)
    research_mask = [True, research_point >= 50, research_point >= 200]

    fuel_rate = [1, 10, 40]
    resoure_ch = [12, 13, 14]
    resoure_map = np.zeros_like(b_active[[12]])

    for resouce_index, mask in enumerate(research_mask):
        if mask:
            resoure_map += (
                b_active[[resoure_ch[resouce_index]]] * fuel_rate[resouce_index]
            )

    resoure_map = resoure_map[
        :, y_shift : input_size[1] - y_shift, x_shift : input_size[0] - x_shift
    ]

    res_avg_map = torch.nn.functional.avg_pool2d(
        input=torch.from_numpy(resoure_map), kernel_size=5, stride=1, padding=2
    ).numpy()

    return res_avg_map, resoure_map


def get_act_cities_map(player_cities: list, game_map: Any):
    # for city in player.cities.values():
    #     for city_tile in city.citytiles:
    #         if city_tile.can_act():
    act_cities_map = np.zeros((1, game_map.height, game_map.width), dtype=np.float32)
    posyx2tile = {}

    for city in player_cities:
        if hasattr(city, "city_cells"):
            for cell in city.city_cells:
                city_tile = cell.city_tile
                if city_tile.can_act():
                    act_cities_map[:, city_tile.pos.y, city_tile.pos.x] = 1.0
                    if city_tile.pos.y in posyx2tile.keys():
                        posyx2tile[city_tile.pos.y].update({city_tile.pos.x: city_tile})
                    else:
                        posyx2tile[city_tile.pos.y] = {city_tile.pos.x: city_tile}

        else:
            for city_tile in city.citytiles:
                if city_tile.can_act():
                    act_cities_map[:, city_tile.pos.y, city_tile.pos.x] = 1.0
                    if city_tile.pos.y in posyx2tile.keys():
                        posyx2tile[city_tile.pos.y].update({city_tile.pos.x: city_tile})
                    else:
                        posyx2tile[city_tile.pos.y] = {city_tile.pos.x: city_tile}

    # act_cities_map_max = torch.nn.functional.max_pool2d(
    #     input=torch.from_numpy(act_cities_map), kernel_size=3, stride=1, padding=1
    # ).numpy()

    return act_cities_map, posyx2tile


def decide_worker_gen_place(
    res_avg_map: np.ndarray,
    act_cities_map_max: np.ndarray,
    resource_map: np.ndarray,
    num_units: int,
    num_city_tiles: int,
):

    places = serarch_with_pooled_feats(res_avg_map, act_cities_map_max)
    if len(places) == 0:
        squeeze_factor = 3
        orig_shape = resource_map.shape[1:]
        kernel_size = (
            resource_map.shape[1] // squeeze_factor,
            resource_map.shape[2] // squeeze_factor,
        )
        pooled_res = skimage.measure.block_reduce(
            resource_map.squeeze(), kernel_size, np.mean
        )
        pooled_res = skimage.transform.resize(
            pooled_res,
            tuple(orig_shape),
            order=None,
            mode="constant",
            clip=True,
            preserve_range=False,
        )

        places = serarch_with_pooled_feats(pooled_res, act_cities_map_max)
    return places[: num_city_tiles - num_units]


def serarch_with_pooled_feats(res_avg_map, act_cities_map_max):
    intersection = (res_avg_map * act_cities_map_max).squeeze()
    places = np.where(intersection > 0)
    if places[0].shape[0] > 0:
        value = intersection[places]
        places = np.stack(places, axis=-1)
        return places[np.argsort(value)[::-1]]
    else:
        return []


def check_action_plan(
    action_code: int,
    our_city: np.ndarray,
    pos_x: int,
    pos_y: int,
    current_plan: np.ndarray,
    is_center: bool = False,
):
    assert type(action_code) == int or np.int64
    is_center = is_center or (action_code == 4)

    use_cooldown_as_center = False
    if is_center:
        pos_y_next = pos_y
        pos_x_next = pos_x
    else:
        direc = ACTION_DIRECTIONS[action_code]
        pos_y_next = direc[1] + pos_y
        pos_x_next = direc[0] + pos_x
        # for random agent
        if (
            (pos_x_next >= our_city.shape[1])
            or (pos_y_next >= our_city.shape[0])
            or (pos_x_next < 0)
            or (pos_y_next < 0)
        ):
            pos_y_next = pos_y
            pos_x_next = pos_x

    is_no_unit = current_plan[pos_y_next, pos_x_next]
    is_citytile = our_city[pos_y_next, pos_x_next]
    is_ok = is_no_unit or is_citytile
    if is_ok:
        if not is_citytile:
            current_plan[pos_y_next, pos_x_next] = False
    else:
        current_plan[pos_y, pos_x] = False
        if not is_center:
            use_cooldown_as_center = True

    return current_plan, use_cooldown_as_center


def check_is_center_action(action_code: int):
    return action_code == 5


def generate_pos_ban_map(b_active: np.ndarray):
    our_units = b_active[3] > 0
    our_city = b_active[8] > 0

    opp_units = b_active[6] > 0
    unit_stack_map = np.logical_or(our_units, opp_units)
    unit_stack_map[our_city] = False

    map_range = b_active[19]
    opp_city = b_active[10]
    ban_map = np.logical_or((map_range == 0), (opp_city > 0))
    ban_map = np.logical_or(ban_map, unit_stack_map)
    return ban_map


class StateHist:
    def __init__(self, input_size: List[int] = [32, 32], unit_hist_length: int = 4):
        """
        additonal histrical input for rl agent
        """
        self.input_size = input_size
        self.unit_hist_length = unit_hist_length
        self.reset()

    def reset(self):
        self.unit_hists = np.zeros(
            [self.unit_hist_length] + self.input_size, dtype=np.float32
        )

        self.initial_resource_map = np.zeros(self.input_size, dtype=np.float32)
        self.resource_reduction_map = np.zeros(self.input_size, dtype=np.float32)
        self.resource_delta = np.zeros(self.input_size, dtype=np.float32)

        self.opponent_unit_track = np.zeros(self.input_size, dtype=np.float32)
        self.player_unit_track = np.zeros(self.input_size, dtype=np.float32)
        self.step = 0

    def initialize(self, initial_state: np.ndarray):
        self.initial_resource_map = self._extract_resource(b_active=initial_state)
        self.resource_reduction_map = self.initial_resource_map / (
            self.initial_resource_map + 1e-6
        )

    def _extract_resource(self, b_active: np.ndarray):
        return b_active[[12, 13, 14]].sum(axis=0)

    def _get_unit_hist(self, b_active: np.ndarray):
        our_units = b_active[2]
        opp_units = b_active[5]
        unit_hist = our_units + opp_units * 0.5
        return our_units, opp_units, unit_hist

    def get_hist(self):
        targets = [
            self.resource_reduction_map,
            self.resource_delta,
            self.player_unit_track,
            self.opponent_unit_track,
        ]
        targets = np.stack(targets, axis=0)
        targets = np.concatenate([self.unit_hists, targets], axis=0)
        return targets[:-2], targets[-2:]

    def update(self, current_state: np.ndarray, env_step: int):
        if self.step == 0:
            self.initialize(initial_state=current_state)

        assert self.step == env_step

        player_units, opp_units, unit_hist = self._get_unit_hist(b_active=current_state)

        self.unit_hists[0:-1] = self.unit_hists[1:]
        self.unit_hists[-1] = unit_hist

        self.opponent_unit_track += opp_units / 360
        self.player_unit_track += player_units / 360
        self.opponent_unit_track = self.opponent_unit_track.clip(0, 1)
        self.player_unit_track = self.player_unit_track.clip(0, 1)

        current_res_stack = (self._extract_resource(b_active=current_state)) / (
            self.initial_resource_map + 1e-6
        )
        self.resource_delta = self.resource_reduction_map - current_res_stack
        self.resource_reduction_map = current_res_stack
        self.step += 1


def not_go_pos(pos_y: int, pos_x: int, ban_map: np.ndarray):
    # n , s, w, e
    # (x, y)
    # action_directions = [(0, -1), (0, +1), (-1, 0), (1, 0)]
    # possible_yx_places = [(direc[1] + pos_y, direc[0] + pos_x) for direc in action_directions]
    action_mask = []
    for direc in ACTION_DIRECTIONS:
        pos_y_next = direc[1] + pos_y
        pos_x_next = direc[0] + pos_x
        if (pos_x_next > 31) or (pos_y_next > 31):
            action_mask.append(False)
        elif (pos_x_next < 0) or (pos_y_next < 0):
            action_mask.append(False)
        else:
            action_mask.append(~ban_map[pos_y_next, pos_x_next])
    return np.array(action_mask).astype(int)
