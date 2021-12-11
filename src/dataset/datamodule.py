import copy
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.dataset import GameObs, LuxDataset, generate_game_obs_from_obs
from src.dataset.utils.process_json import (
    convert_samples_in_df,
    create_dataset_from_json,
)

IMG_MEAN = (0.485, 0.456, 0.406) * 5
IMG_STD = (0.229, 0.224, 0.225) * 5


class LuxDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        conf: DictConfig,
        batch_size: int = 64,
        num_workers: int = 16,
        aug_mode: int = 0,
        is_debug: bool = False,
    ) -> None:
        super().__init__()
        self.conf = conf
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.num_workers = num_workers
        self.is_debug = is_debug
        self.num_inchannels = 3 * 3  # site num * abs, cos, sin

        self.img_mean = np.array(IMG_MEAN[: self.num_inchannels])
        self.img_std = np.array(IMG_STD[: self.num_inchannels])

    def prepare_data(self):
        # check
        assert Path(get_original_cwd(), self.conf["data_dir"]).is_dir()

    def _onehot_to_set(self, onehot: np.ndarray):
        return set(np.where(onehot == 1)[0].astype(str).tolist())

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders

        conf = self.conf
        if stage == "fit" or stage is None:
            # for hydra
            cwd = get_original_cwd()

            # load data
            episode_dir = Path(cwd, self.conf["data_dir"])
            obses, samples, action_hists = create_dataset_from_json(episode_dir)
            print("obses:", len(obses), "samples:", len(samples))

            game_obses = convert_obs_into_game(
                obses=obses,
                num_state_features=conf.obs.num_state_features,
                input_size=tuple(conf.obs.input_size),
                cache_dir=Path(cwd, conf.obs.input_cache_dir),
                num_workers=conf.num_workers,
            )

            labels = [sample[-1] for sample in samples]
            actions = ["north", "south", "west", "east", "bcity"]
            for value, count in zip(*np.unique(labels, return_counts=True)):
                print(f"{actions[value]:^5}: {count:>3}")

            obses, self.train_df = convert_samples_in_df(obses=obses, samples=samples)
            # train/val split
            self.train_df = make_split(
                df=self.train_df,
                n_splits=conf.n_splits,
                target_key=conf.target_key,
                group_key=conf.group_key,
                how=conf.split_how,
            )
            if conf.model.type == "image_caption":
                self.train_df = self.train_df[
                    ~self.train_df.duplicated("obs_id", keep="first")
                ]

            train_df = self.train_df.loc[self.train_df["fold"] != conf.val_fold, :]
            val_df = self.train_df.loc[self.train_df["fold"] == conf.val_fold, :]


            self.train_dataset = LuxDataset(
                obses=obses,
                game_obses=game_obses,
                samples=train_df[["obs_id", "unit_id", "target"]].to_numpy(),
                num_state_features=conf.obs.num_state_features,
                input_size=tuple(conf.obs.input_size),
                is_xy_order=conf.obs.is_xy_order,
                cache_obs_in_memory=conf.obs.cache_obs_in_memory,
                transforms=self.train_transform(),
                random_crop=conf.obs.random_crop,
                action_hists=action_hists,
                decoder_in_features=conf.model.decoder.in_features,
                ignore_class_index=conf.model.ignore_class_index,
                max_sequence=conf.model.decoder.max_sequence,
                skip_active_unit_drawing=conf.model.type == "image_caption",
                no_action_index=conf.model.no_action_index,
                random_start_ordering=True,
                no_action_droprate=conf.model.no_action_droprate
            )
            self.val_dataset = LuxDataset(
                obses=obses,
                game_obses=game_obses,
                samples=val_df[["obs_id", "unit_id", "target"]].to_numpy(),
                num_state_features=conf.obs.num_state_features,
                input_size=tuple(conf.obs.input_size),
                is_xy_order=conf.obs.is_xy_order,
                cache_obs_in_memory=conf.obs.cache_obs_in_memory,
                transforms=None,
                random_crop=False,
                action_hists=action_hists,
                decoder_in_features=conf.model.decoder.in_features,
                ignore_class_index=conf.model.ignore_class_index,
                max_sequence=conf.model.decoder.max_sequence,
                skip_active_unit_drawing=conf.model.type == "image_caption",
                no_action_index=conf.model.no_action_index,
                random_start_ordering=False,
                no_action_droprate=0.0 if conf.model.no_action_droprate != 1.0 else 1.0,
            )

            self.plot_dataset(self.train_dataset)
            self.train_df = train_df
            self.val_df = val_df

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_transform(self):
        return self.get_transforms(mode=self.aug_mode)

    def val_transform(self):
        return self.get_transforms(mode=0)

    def test_transform(self):
        return self.get_transforms(mode=0)

    def get_transforms(self, mode: int = 0) -> albu.Compose:

        if mode == 0:
            # transforms = [
            #     # albu.Lambda(image=add_pad_img, mask=add_pad_mask, name="padding"),
            #     albu.Normalize(mean=self.img_mean, std=self.img_std),
            # ]
            return None
        elif mode == 1:
            transforms = [
                albu.Flip(p=0.75),
                albu.RandomRotate90(p=0.5),
            ]
        else:
            raise NotImplementedError
        # if self.conf.gt_as_mask:
        #     additional_targets = {"target_image": "mask"}
        # else:
        #     additional_targets = {"target_image": "image"}

        # composed = albu.Compose(transforms, additional_targets=additional_targets)
        # return composed
        return albu.Compose(transforms)

    def plot_dataset(
        self,
        dataset,
        plot_num: int = 3,
        df: Optional[pd.DataFrame] = None,
        use_clear_event: bool = True,
    ) -> None:
        inds = np.random.choice(len(dataset), plot_num)

        for ind in inds:
            data = dataset[ind]
            im = data["image"].numpy().transpose(1, 2, 0)
            # === PLOT ===
            nrows = 2
            ncols = 4
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(12, 6),
                sharey=False,
                sharex=False,
            )
            ch_start = 0
            for i in range(nrows):
                for j in range(ncols):
                    if ch_start > im.shape[-1]:
                        continue
                    ch_end = ch_start + 3
                    ch_start = min(ch_end - 3, im.shape[-1] - 3)
                    ax[i][j].imshow(im[:, :, ch_start:ch_end])
                    ch_start += 3

            title = f'id: {data["id"]}' + " " + f'unit_id: {data["unit_id"]}'
            has_value = isinstance(data["target"].numpy().tolist(), int)
            if has_value:
                title += " target: " + str(data["target"].numpy().tolist())

            fig.suptitle(title)


def make_split(
    df: pd.DataFrame,
    n_splits: int = 3,
    target_key: str = "target",
    group_key: Optional[str] = None,
    is_reset_index: bool = True,
    verbose: int = 1,
    shuffle: bool = True,
    how: str = "stratified",
) -> pd.DataFrame:

    if shuffle:
        df = df.sample(frac=1.0)

    if is_reset_index:
        df.reset_index(drop=True, inplace=True)
    df["fold"] = -1

    split_keys = {"X": df, "y": df[target_key]}
    if how == "stratified":
        cv = StratifiedKFold(n_splits=n_splits)
    elif how == "group":
        assert group_key is not None
        cv = GroupKFold(n_splits=n_splits)
        split_keys.update({"groups": df[group_key]})
    elif how == "stratified_group":
        assert group_key is not None
        cv = StratifiedGroupKFold(n_splits=n_splits)
        split_keys.update({"groups": df[group_key]})
    else:
        raise ValueError(f"how: {how}")

    for i, (train_idx, valid_idx) in enumerate(cv.split(**split_keys)):
        df.loc[valid_idx, "fold"] = i
    if verbose == 1:
        print(">> check split with target\n", pd.crosstab(df.fold, df[target_key]))
        if group_key is not None:
            print(">> check split with group\n", pd.crosstab(df.fold, df[group_key]))

    return df


def convert_obs_into_game(
    obses: dict,
    num_state_features: int,
    input_size: list,
    cache_dir: Optional[Path] = None,
    num_workers: int = 16,
) -> Dict[str, Union[GameObs, Path]]:

    if cache_dir is not None:
        cache_paths = list(cache_dir.glob("*.pickle"))
        cache_obs_ids = [cache_path.stem for cache_path in cache_paths]
        if set(cache_obs_ids) >= set(obses.keys()):
            return dict(zip(cache_obs_ids, cache_paths))

    # game_obses: Dict[str, Union[GameObs, Path]] = {}
    # for obs_id, obs in tqdm(obses.items(), total=len(obses)):
    #     obs["obs_id"] = obs_id
    #     game_obses[obs_id] = generate_game_obs_from_obs(
    #         obs=obs,
    #         num_state_features=num_state_features,
    #         input_size=tuple(input_size),
    #         cache_dir=cache_dir,
    #     )
    obes_for_para = copy.deepcopy(obses)
    for obs_id, obs in obes_for_para.items():
        obs["obs_id"] = obs_id

    func_ = partial(
        generate_game_obs_from_obs,
        num_state_features=num_state_features,
        input_size=tuple(input_size),
        cache_dir=cache_dir,
    )
    with Pool(processes=num_workers) as pool:
        # game_obses = pool.map(func_, obes_for_para.values())
        cache_paths = list(
            tqdm(pool.imap(func_, obes_for_para.values()), total=len(obes_for_para))
        )

    cache_obs_ids = [cache_path.stem for cache_path in cache_paths]
    return dict(zip(cache_obs_ids, cache_paths))
