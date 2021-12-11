from typing import Dict, OrderedDict

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, pooled_feature: int = 128
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.value_projection = nn.Sequential(
            OrderedDict(
                [
                    ("value_bn", nn.BatchNorm1d(pooled_feature)),
                    ("value_relu", nn.ReLU()),
                    ("value_fc", nn.Linear(pooled_feature, 1)),
                ]
            )
        )
        self.pooled_feature = pooled_feature

    def forward(self, x: Dict[str, torch.Tensor], _=None):
        encoder_out = self.encoder(x["image"])
        decoder_out = self.decoder(x["input_sequence"], encoder_out)
        (
            bsz,
            _,
            _,
            _,
        ) = x["image"].size()
        pooled = encoder_out["pooled"].view(bsz, self.pooled_feature)
        value = self.value_projection(pooled)
        return {"policy": decoder_out["outputs"], "value": value}
