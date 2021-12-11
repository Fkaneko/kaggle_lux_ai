# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import random

from LuxPythonEnvGym.build.lib.luxai2021.game import unit


def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


def bimap_r(x, y, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(bimap_r(xx, y[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, bimap_r(xx, y[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y) if callback_fn is not None else None


def trimap_r(x, y, z, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(trimap_r(xx, y[i], z[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, trimap_r(xx, y[key], z[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y, z) if callback_fn is not None else None


def rotate(x, max_depth=1024):
    if max_depth == 0:
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(
                rotate(type(x)(xx[i] for xx in x), max_depth - 1)
                for i, _ in enumerate(x[0])
            )
        elif isinstance(x[0], dict):
            return type(x[0])(
                (key, rotate(type(x)(xx[key] for xx in x), max_depth - 1))
                for key in x[0]
            )
    elif isinstance(x, dict):
        x_front = x[list(x.keys())[0]]
        if isinstance(x_front, (list, tuple)):
            return type(x_front)(
                rotate(type(x)((key, xx[i]) for key, xx in x.items()), max_depth - 1)
                for i, _ in enumerate(x_front)
            )
        elif isinstance(x_front, dict):
            return type(x_front)(
                (key2, rotate(type(x)((key1, xx[key2]) for key1, xx in x.items()), max_depth - 1))
                for key2 in x_front
            )
    return x


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def get_random_action(obs, legal_actions):
    sequence_mask = obs.get("sequence_mask", None)
    if sequence_mask is None:
        # TODO: which one [0] or wo [0]?
        # return random.choice(legal_actions)
        return random.choice(legal_actions)[0]
    else:
        action = np.zeros_like(sequence_mask.squeeze())
        unit_seq = np.where(sequence_mask > 0)[0]
        action_mask = obs["rule_mask"] != 0
        for unit_ind in unit_seq:
            acts = np.array(legal_actions)[action_mask[unit_ind]]
            if acts.shape[0] == 0:
                acts = legal_actions
            action[unit_ind] = random.choices(acts)[0]
        return action


def get_action_code(obs, policy, legal_actions, temperature=0, mode="g"):
    sequence_mask = obs.get("sequence_mask", None)
    action_mask = np.ones_like(policy) * 1e32
    if sequence_mask is None:
        action_mask[legal_actions] = 0
    else:
        if sequence_mask.sum() == 0:
            # no actionable units
            return None, None, None,
        action_mask[:, legal_actions] = 0
        action_mask[obs["rule_mask"] == 0] = 1e32

    policy = policy - action_mask
    action = choose_action(
        legal_actions=legal_actions, policy=policy, temperature=temperature, sequence_mask=sequence_mask, mode=mode
    )

    return action_mask, policy, action

def choose_action(legal_actions, policy, temperature=0, sequence_mask=None, mode="g"):
    if sequence_mask is None:
        if mode == "g":
            action = random.choices(legal_actions, weights=softmax(policy[legal_actions]))[0]
        elif mode == "e":
            if temperature == 0:
                # choose the heighest proba action
                ap_list = sorted([(a, policy[a]) for a in legal_actions], key=lambda x: -x[1])
                action = ap_list[0][0]
            else:
                action = random.choices(np.arange(len(policy)), weights=softmax(policy / temperature))[0]
    else:
        action = np.zeros_like(sequence_mask.squeeze())
        unit_seq = np.where(sequence_mask > 0)[0]
        if mode == "g":
            for unit_ind in unit_seq:
                action[unit_ind] = random.choices(legal_actions, weights=softmax(policy[unit_ind][legal_actions]))[0]
        elif mode == "e":
            if temperature == 0:
                # choose the heighest proba action
                for unit_ind in unit_seq:
                    ap_list = sorted([(a, policy[unit_ind][a]) for a in legal_actions], key=lambda x: -x[1])
                    action[unit_ind] = ap_list[0][0]
            else:
                for unit_ind in unit_seq:
                    ap_list = sorted([(a, policy[unit_ind][a]) for a in legal_actions], key=lambda x: -x[1])
                    action[unit_ind] = random.choices(np.arange(len(policy[unit_ind])), weights=softmax(policy[unit_ind] / temperature))[0]
    return action
