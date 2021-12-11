import math
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from torchvision.transforms import Normalize
from torchvision.utils import make_grid

from src.modeling.model_arch.build_model import build_model_from_conf
from src.rl.internal_validation import (
    ImitationAgent,
    internal_match,
    load_baseline_model,
    load_opponent,
    vis_match_results,
)


class LitModel(pl.LightningModule):
    def __init__(
        self, conf: DictConfig, dataset_len: int = 72899, logger_name="tensorboard"
    ) -> None:
        super().__init__()
        self.conf = conf
        self.dataset_len = dataset_len
        self.logger_name = logger_name

        net_kwargs, net_class = build_model_from_conf(conf=self.conf)

        loss_reduction = "mean"
        self.num_inchannels = conf.model[conf.model.type].in_channels

        self.model = net_class(**net_kwargs)
        if (self.conf.model.type == "image_caption") & (
            net_kwargs["encoder"] == "imagenet"
        ):
            patch_first_conv(
                self.model,
                in_channels=self.num_inchannels,
                stride_override=self.conf.model.stride_override,
            )

        if self.conf.model.channels_last:
            # Need to be done once, after model initialization (or load)
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.conf.model.loss.type == "mse":
            self.criterion = torch.nn.MSELoss(reduction=self.conf.model.loss.reduction)
        elif self.conf.model.loss.type == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss(
                reduction=self.conf.model.loss.reduction
            )
        elif self.conf.model.loss.type == "ce":
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=self.conf.model.loss.reduction
            )
        else:
            raise NotImplementedError

        if self.conf.model.metric == "f1":
            self.metric = torchmetrics.F1()
        else:
            raise NotImplementedError
        self.metrics_fn = {"outputs": self.metric, "aux_out": None}

        self.aux_criterion = torch.nn.CrossEntropyLoss(
            reduction=self.conf.model.loss.reduction
        )
        self.criterions = {
            "outputs": self.criterion,
            "aux_out": self.aux_criterion,
        }
        self.loss_weights = self.conf.model.loss.weights

        if self.conf.model.last_act == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif self.conf.model.last_act == "tanh":
            self.activation = torch.nn.Tanh()
        elif self.conf.model.last_act == "identity":
            self.activation = torch.nn.Identity()
        elif self.conf.model.last_act == "softmax":
            self.activation = torch.nn.Softmax(dim=-1)
        else:
            raise NotImplementedError
        self.aux_activation = torch.nn.Softmax(dim=-1)
        self.activations = {
            "outputs": self.activation,
            "aux_out": self.aux_activation,
        }

        self.val_sync_dist = self.conf.trainer.gpus > 1
        self.is_debug = self.conf.is_debug

        self.pred_df = pd.DataFrame({})

        try:
            cwd = get_original_cwd()
        except:
            cwd = os.getcwd()

        opponent_path = os.path.join(
            cwd,
            "../input/lux_ai_baseline_imitation_weight",
        )
        # self.opponent_model = load_baseline_model(path=opponent_path)
        # self.opponent = ImitationAgent(
        #     mode="pred", model=self.opponent_model, is_xy_order=True, is_cuda=False
        # )

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def show(imgs: torch.Tensor):
        """
        for tensor debugging
        imgs (ba, ch, h, w) shape
        """
        import torchvision.transforms.functional as F

        imgs = make_grid(imgs)

        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    @staticmethod
    def process_multi_outputs(
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        activations: Dict[str, torch.nn.Module],
        criterions: Dict[str, torch.nn.Module],
        metrics_fn: Optional[Dict[str, torch.nn.Module]] = None,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        preds = {}
        losses = {}
        metrics = {}
        for output_key, output in outputs.items():
            if output is None:
                continue
            preds[output_key] = activations[output_key](output).squeeze()
            if sequence_mask is not None:
                mask = (sequence_mask == 1).squeeze()
                preds[output_key] = preds[output_key].max(dim=-1)[1][mask]
                # bs, T, class -> bs, class, T
                output = output.squeeze().transpose(2, 1)
                losses[output_key] = (
                    criterions[output_key](output, targets.squeeze()) * mask
                ).sum() / mask.sum()
            else:
                losses[output_key] = criterions[output_key](
                    output.squeeze(), targets.squeeze()
                )
            if metrics_fn is not None:
                if sequence_mask is not None:
                    metrics[output_key] = metrics_fn[output_key](
                        preds[output_key],
                        targets[mask],
                    )
                else:
                    metrics[output_key] = metrics_fn[output_key](
                        preds[output_key],
                        targets,
                    )

        return preds, losses, metrics

    @staticmethod
    def logging_multi_outputs(
        log_targets: Dict[str, torch.Tensor],
        logger: Callable,
        log_name: str = "loss",
        log_header: str = "train",
        logger_name: str = "tensorboard",
    ) -> None:
        name_for_logger = f"{log_header}/{log_name}"
        for output_key, log_target in log_targets.items():
            if output_key == "outputs":
                if logger_name == "tensorboard":
                    logger(name_for_logger, log_target)
                elif logger_name == "neptune":
                    logger.experiment[name_for_logger].log(log_target)
            elif log_target is not None:
                if logger_name == "neptune":
                    logger.experiment[f"{name_for_logger}_{output_key}"].log(log_target)

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.conf.model.channels_last:
            # Need to be done for every input
            inputs = inputs.to(memory_format=torch.channels_last)

        # targets = batch["target"]
        targets = batch["output_sequence"]
        sequence_mask = batch.get("sequence_mask", None)

        # outputs is dict for multi heads
        outputs = self.model(inputs, aux_inputs=batch["input_sequence"])
        preds, losses, metrics = LitModel.process_multi_outputs(
            outputs=outputs,
            targets=targets,
            activations=self.activations,
            criterions=self.criterions,
            metrics_fn=self.metrics_fn,
            sequence_mask=sequence_mask,
        )
        if sequence_mask is not None:
            targets = targets[(sequence_mask == 1)]

        for log_name, log_target in [
            ("loss", losses),
            (self.conf.model.metric, metrics),
        ]:
            LitModel.logging_multi_outputs(
                log_targets=log_target,
                logger=self.logger if self.logger_name == "neptune" else self.log,
                log_name=log_name,
                log_header="train",
                logger_name=self.logger_name,
            )

        total_loss = 0
        for output_key, loss in losses.items():
            if loss is None:
                continue
            total_loss += self.loss_weights[output_key] * loss

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.conf.model.channels_last:
            # Need to be done for every input
            inputs = inputs.to(memory_format=torch.channels_last)

        # targets = batch["target"]
        targets = batch["output_sequence"]
        sequence_mask = batch.get("sequence_mask", None)

        # outputs is dict for multi heads
        outputs = self.model(inputs, aux_inputs=batch["input_sequence"])
        preds, losses, metrics = LitModel.process_multi_outputs(
            outputs=outputs,
            targets=targets,
            activations=self.activations,
            criterions=self.criterions,
            metrics_fn=self.metrics_fn,
            sequence_mask=sequence_mask,
        )
        if sequence_mask is not None:
            targets = targets[(sequence_mask == 1)]

        for log_name, log_target in [
            ("loss", losses),
            (self.conf.model.metric, metrics),
        ]:
            LitModel.logging_multi_outputs(
                log_targets=log_target,
                logger=self.logger if self.logger_name == "neptune" else self.log,
                log_name=log_name,
                log_header="val",
                logger_name=self.logger_name,
            )

        if self.logger_name == "neptune":
            if self.conf.monitor == "val_win_rate":
                pass
            elif self.conf.monitor == "val_loss":
                self.log(self.conf.monitor, losses["outputs"])
            elif self.conf.monitor == f"val_{self.conf.model.metric}":
                self.log(self.conf.monitor, metrics["outputs"])

        return {"id": batch["id"], "pred": preds["outputs"], "targets": targets}

    @staticmethod
    def parse_batched_outputs(batched_outputs: list):
        keys = list(batched_outputs[0].keys())
        met_dict = {key: [] for key in keys}
        for pred_batch in batched_outputs:
            for key in keys:
                met_dict[key].append(pred_batch[key])

        for key in keys:
            if isinstance(met_dict[key][0], torch.Tensor):
                met_dict[key] = (
                    torch.cat(met_dict[key]).cpu().numpy().astype(np.float32)
                )

            elif isinstance(met_dict[key][0], np.ndarray):
                met_dict[key] = np.concatenate(met_dict[key])

            elif isinstance(met_dict[key][0], list):
                met_dict[key] = np.concatenate(met_dict[key])

            else:
                raise ValueError(f"unexpected type {type(met_dict[key])}")

        return met_dict

    def _logging_fig(self, fig, log_name: str) -> None:
        if self.logger_name == "tensorboard":
            self.logger.experiment.add_figure(
                log_name,
                fig,
                global_step=self.trainer.global_step,
            )
        elif self.logger_name == "neptune":
            self.logger.experiment[log_name].log(fig)

    def _logging_value(self, value: Union[float, int], log_name: str) -> None:
        if self.logger_name == "tensorboard":
            self.log(log_name, value)
        elif self.logger_name == "neptune":
            self.logger.experiment[log_name].log(value)

    def _mask_ignore_class_index(self, pred: np.ndarray, targets: np.ndarray):
        # mask_ = targets != self.ignore_class_index
        # return pred[mask_], targets[mask_]
        return pred, targets

    def validation_epoch_end(self, val_step_outputs):
        met_dict = LitModel.parse_batched_outputs(batched_outputs=val_step_outputs)

        targets = met_dict["targets"]
        pred = met_dict["pred"]
        pred, targets = self._mask_ignore_class_index(pred=pred, targets=targets)

        accuracy = accuracy_score(y_true=targets, y_pred=pred)
        self._logging_value(value=accuracy, log_name="val/acc")

        # player = ImitationAgent(
        #     mode="pred",
        #     model=self.model,
        #     is_xy_order=self.conf.obs.is_xy_order,
        #     is_cuda=False,
        # )
        # num_episodes = 10 if self.conf.is_debug else self.conf.internal_val.num_episodes

        # replay_folder = None
        # if self.conf.model.save_replay:
        #     replay_folder = f"./replay_jsons/{self.trainer.global_step}"

        # match_results = internal_match(
        #     player=player,
        #     opponent=self.opponent,
        #     num_episodes=num_episodes,
        #     replay_folder=replay_folder,
        #     replay_stateful=False,
        # )
        # for score_name, res in match_results.items():
        #     if score_name == "scores":
        #         fig, _ = vis_match_results(scores=res)
        #         self._logging_fig(fig=fig, log_name="val/score_hist")
        #         continue
        #     self._logging_value(value=res, log_name=f"val/{score_name}")
        # plt.close()

        # self.log("val_win_rate", match_results["win_rate"])

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if not self.conf.find_lr:
            if self.trainer.global_step < self.warmup_steps:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.warmup_steps
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.conf.lr
            else:
                pct = (self.trainer.global_step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                pct = min(1.0, pct)
                for pg in optimizer.param_groups:
                    pg["lr"] = self._annealing_cos(pct, start=self.conf.lr, end=0.0)

        if self.logger_name == "neptune":
            self.logger.experiment["train/lr"].log(optimizer.param_groups[0]["lr"])
        optimizer.step(closure=closure)
        optimizer.zero_grad()

    def _annealing_cos(self, pct: float, start: float = 0.1, end: float = 0.0) -> float:
        """
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
        Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.
        """
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def configure_optimizers(self):
        self.total_steps = (
            self.dataset_len // self.conf.batch_size
        ) * self.conf.trainer.max_epochs
        self.warmup_steps = int(self.total_steps * self.conf.warmup_ratio)

        if self.conf.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.conf.lr,
                momentum=0.9,
                weight_decay=4e-5,
            )
        elif self.conf.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        else:
            raise NotImplementedError
        # steps_per_epoch = self.hparams.dataset_len // self.hparams.batch_size
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.lr,
        #     max_epochs=self.hparams.max_epochs,
        #     steps_per_epoch=steps_per_epoch,
        # )
        # return [optimizer], [scheduler]
        return optimizer


def patch_first_conv(
    model, in_channels: int = 4, stride_override: Optional[Tuple[int, int]] = None
) -> None:
    """
    from segmentation_models_pytorch/encoders/_utils.py
    Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    # reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    elif in_channels == 3:
        pass
    elif in_channels == 4:
        weight = torch.nn.Parameter(torch.cat([weight, weight[:, -1:, :, :]], dim=1))
    elif in_channels % 3 == 0:
        weight = torch.nn.Parameter(torch.cat([weight] * (in_channels // 3), dim=1))

    module.weight = weight
    if stride_override is not None:
        assert module.stride == (2, 2), "wrong stride_override target"
        module.stride = tuple(stride_override)
