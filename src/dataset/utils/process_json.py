import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from luxai2021.game.constants import Constants, LuxMatchConfigs_Default
from tqdm import tqdm

ACTIONS = Constants.ACTIONS
DIRECTIONS = Constants.DIRECTIONS

# DIRECTION_LABELS = {"c": None, "n": 0, "s": 1, "w": 2, "e": 3}
DIRECTION_LABELS = {
    DIRECTIONS.CENTER: None,
    DIRECTIONS.NORTH: 0,
    DIRECTIONS.SOUTH: 1,
    DIRECTIONS.WEST: 2,
    DIRECTIONS.EAST: 3,
}
ALL_LABELS = {
    ACTIONS.MOVE: 0,
    ACTIONS.BUILD_CITY: 4,
    ACTIONS.RESEARCH: 5,
    ACTIONS.BUILD_WORKER: 6,
    ACTIONS.BUILD_CART: 7,
    ACTIONS.PILLAGE: 8,
    ACTIONS.TRANSFER: 9,
}


def to_label(action: str, with_all_actions: bool = False):
    """
    from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    """

    def handle_reamining_actions(action_strs: List[str]) -> int:
        if action_strs[0] == ACTIONS.TRANSFER:
            # label = ALL_LABELS[ACTIONS.BUILD_CITY] + DIRECTION_LABELS[action_strs[-1]]
            label = ALL_LABELS[ACTIONS.TRANSFER]
        else:
            label = ALL_LABELS[action_strs[0]]
        return label

    strs = action.split(" ")
    unit_id = strs[1]
    if strs[0] == ACTIONS.MOVE:
        label = DIRECTION_LABELS[strs[2]]
    elif strs[0] == ACTIONS.BUILD_CITY:
        label = ALL_LABELS[ACTIONS.BUILD_CITY]
    else:
        if not with_all_actions:
            label = None
        else:
            label = handle_reamining_actions(action_strs=strs)

    return unit_id, label


def label_to_action(label: int) -> str:
    l2a_dict = {
        value: key for key, value in DIRECTION_LABELS.items() if value is not None
    }
    return l2a_dict[label]


def depleted_resources(obs):
    """
    from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    """
    for u in obs["updates"]:
        if u.split(" ")[0] == "r":
            return False
    return True


def create_dataset_from_json(
    episode_dir: Path,
    team_name: str = "Toad Brigade",
    team_index: Optional[str] = None,
    episode_id: Optional[int] = None,
    skip_zero_resource: bool = True,
) -> Tuple[Dict[str, Any], List[Any], Dict[str, List[List[str]]]]:
    """
    from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
    team_name: target team name to imitate, which will be converted into 0 or 1 at json

    input is obsevation of time T.
    label is action of time T+1 with observation of time T.

    labeling 4 move action and city building until all resources are consumed.
    """
    obses = {}
    samples = []
    action_strs = defaultdict(list)
    target_obs_keys = [
        "step",
        "updates",
        "player",
        "width",
        "height",
        "globalCityIDCount",
        "globalUnitIDCount",
    ]
    if episode_id is not None:
        episodes = [Path(episode_dir, str(episode_id) + ".json")]
    else:
        episodes = [
            path
            for path in Path(episode_dir).glob("*.json")
            if "info" not in path.name
        ]
    for filepath in tqdm(episodes):
        with open(filepath) as f:
            json_load = json.load(f)

        newpath = Path(filepath.parent, filepath.name.replace("episodes", ""))
        if not newpath.exists():
            filepath = filepath.rename(newpath)

        ep_id = json_load["info"]["EpisodeId"]

        if team_index is not None:
            index = team_index
        else:
            # choose win team with team name
            index = np.argmax([r or 0 for r in json_load["rewards"]])
            # team name is conveted into index 0 or 1
            if json_load["info"]["TeamNames"][index] != team_name:
                continue

        for i in range(len(json_load["steps"]) - 1):
            if json_load["steps"][i][index]["status"] == "ACTIVE":
                # time step is different, input and label relation
                # obs is always plaed at player 0 side json
                actions = json_load["steps"][i + 1][index]["action"]
                obs = json_load["steps"][i][0]["observation"]
                # resorce check
                if skip_zero_resource and depleted_resources(obs):
                    break

                obs["player"] = index
                obs = dict([(k, v) for k, v in obs.items() if k in target_obs_keys])
                obs_id = f"{ep_id}_{i}"
                obses[obs_id] = obs

                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None:
                        samples.append((obs_id, unit_id, label))

                actions_0 = json_load["steps"][i + 1][0]["action"]
                actions_1 = json_load["steps"][i + 1][1]["action"]
                action_strs[ep_id].append(
                    [
                        actions_0 if actions_0 is not None else [],
                        actions_1 if actions_1 is not None else [],
                    ]
                )

    return obses, samples, action_strs


def convert_samples_in_df(obses: dict, samples: list) -> pd.DataFrame:
    df = pd.DataFrame(samples, columns=["obs_id", "unit_id", "target"])
    df["episode_id"] = df["obs_id"].apply(lambda x: str(x.split("_")[0]))
    return obses, df
