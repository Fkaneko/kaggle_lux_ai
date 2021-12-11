import torch.nn.functional as F
from torch import nn


# Neural Network for Lux AI
class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        """
        from https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning
        """
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h
