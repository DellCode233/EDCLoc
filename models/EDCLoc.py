import torch.nn as nn
import torch
from models.Skipconnect import SkipConnect


class slot_conv(nn.Module):
    def __init__(self, basesize, tmpsize, kernel_size, groups) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.15),
            nn.Conv1d(basesize, tmpsize, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv1d(tmpsize, basesize, kernel_size=kernel_size, padding="same", bias=False, groups=groups),
        )

    def forward(self, X):
        return self.net(X)


class slot_block(nn.Module):
    def __init__(self, basesize, tmpsize, kernel_size, groups) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SkipConnect(
                # nn.Sequential(
                #     SkipConnect(slot_conv(basesize, tmpsize, kernel_size, groups)),
                #     nn.ReLU(),
                #     SkipConnect(slot_conv(basesize, tmpsize, kernel_size, groups)),
                # )
                nn.Sequential(
                    nn.Dropout(0.15),
                    nn.Conv1d(basesize, tmpsize, 1, bias=False),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Conv1d(tmpsize, basesize, kernel_size=kernel_size, padding="same", bias=False, groups=groups),
                )
            ),
            nn.ReLU(),
        )

    def forward(self, X):
        return self.net(X)


class slot_head(nn.Module):
    def __init__(self, out_channels, kernels=[9, 21, 39, 49, 59], hidden_channels=16) -> None:
        super().__init__()
        self.convlist = nn.ModuleList(
            [
                nn.Conv1d(5, hidden_channels, kernel_size=kernels[4], padding="same", bias=False),
                nn.Conv1d(5, hidden_channels, kernel_size=kernels[3], padding="same", bias=False),
                nn.Conv1d(5, hidden_channels, kernel_size=kernels[2], padding="same", bias=False),
                nn.Conv1d(5, hidden_channels, kernel_size=kernels[1], padding="same", bias=False),
                nn.Conv1d(5, hidden_channels, kernel_size=kernels[0], padding="same", bias=False),
            ]
        )
        self.conv1x1 = nn.Conv1d(hidden_channels * 5, out_channels, kernel_size=1, bias=True)
        # self.drop = nn.Dropout(0.05)

    def forward(self, X):
        out = torch.concat([module(X) for module in self.convlist], dim=1)
        return self.conv1x1(out)


class slot_adaptive_pool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # B C S
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, X):
        return self.pool(X)


class classifier(nn.Module):
    def __init__(
        self,
        h_kernels,
        h_channels,
        mpool0=8,
        tmpsize=96,
        basesize=64,
        groups=8,
        kernels=[96, 49, 36, 21, 5],
        poolsize=2,
    ) -> None:
        """
        # B x C x S
        """
        super().__init__()
        self.block_handle = nn.Sequential(
            # nn.Conv1d(5, basesize, kernel_size=kernels[0], padding="same", bias=False),
            slot_head(
                basesize,
                h_kernels,
                h_channels,
            ),
            nn.ReLU(),
            nn.MaxPool1d(mpool0, mpool0),
            slot_block(basesize, tmpsize, kernels[0], groups),
            nn.MaxPool1d(poolsize, poolsize),
            slot_block(basesize, tmpsize, kernels[1], groups),
            nn.MaxPool1d(poolsize, poolsize),
            slot_block(basesize, tmpsize, kernels[2], groups),
            nn.MaxPool1d(poolsize, poolsize),
            slot_block(basesize, tmpsize, kernels[3], groups),
            nn.MaxPool1d(poolsize, poolsize),
            slot_block(basesize, tmpsize, kernels[4], groups),
        )
        self.global_pool = slot_adaptive_pool()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0),
            nn.Linear(basesize, 100),
            nn.Hardswish(),
            nn.Dropout(),
            nn.Linear(100, 6),
        )

    def forward(self, X):
        return self.MLP(self.global_pool(self.block_handle(X.permute(0, 2, 1))))

    @staticmethod
    def get_model_params():
        model_params = dict()
        model_params["mpool0"] = 8
        model_params["tmpsize"] = 96
        model_params["basesize"] = 64
        model_params["groups"] = 8
        model_params["kernels"] = [96, 49, 36, 21, 5]
        model_params["poolsize"] = 2
        diff = 9  # 7
        c0 = 9
        model_params["h_kernels"] = [c0, c0 + diff, c0 + 2 * diff, c0 + 3 * diff, c0 + 4 * diff]
        model_params["h_channels"] = 20  # 20
        return model_params

    @staticmethod
    def get_hparams():
        hparams = dict(lr=1e-4, batchsize=64, num_labels=6, max_epochs=400, task="multilabel")
        hparams["name"] = f"EDCLoc"
        hparams["lr"] = 1e-4
        hparams["clip_algo"] = "norm"
        hparams["clip_value"] = 0.5
        hparams["early_stopping"] = {"patience": 40, "monitor": "valid_S", "mode": "max"}
        hparams["ckpts"] = [dict(monitor=f"Ss/S{idx}", mode="max") for idx in range(6)]
        return hparams
