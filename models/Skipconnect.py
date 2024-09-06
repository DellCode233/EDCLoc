import torch.nn as nn


class SkipConnect(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.net = module

    def forward(self, X):
        return X + self.net(X)
