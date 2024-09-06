import warnings

warnings.filterwarnings("ignore")
from typing import Any
import torch
import torch.nn as nn
import os
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    F1Score,
    Precision,
    Recall,
    ConfusionMatrix,
)
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, OptimizerLRScheduler


def get_MCCs(matrixs):
    total = []
    for matrix in matrixs:
        TP, FN, FP, TN = matrix[[1, 1, 0, 0], [1, 0, 1, 0]]
        numerator = (TP * TN) - (FP * FN)
        denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        if denominator <= 0.0:
            total.append(0.0)
        else:
            total.append(numerator / (denominator**0.5))
    return torch.tensor(total)


class LitAuto(pl.LightningModule):
    def __init__(self, module, model_params, task, lr=1e-4, weight_decay=0, **args) -> None:
        super().__init__()
        # task argument to either 'binary', 'multiclass' or 'multilabel'
        self.module = module(**model_params) if isinstance(module, type) else module
        self.save_hyperparameters(ignore="module")
        if task == "binary":
            metric_param = dict(task=task)
        elif task == "multilabel":
            metric_param = dict(task=task, num_labels=self.hparams.num_labels)
        else:
            metric_param = dict(task=task, num_classes=self.hparams.num_classes)

        self.loss_func = self.hparams.get("loss_func", None)
        self.loss_func_hparams = self.hparams.get("loss_func_hparams", {})
        if self.loss_func is None:
            self.loss_func = nn.functional.binary_cross_entropy_with_logits
        self.train_acc = Accuracy(**metric_param)
        # -----------------------------
        self.valid_acc = Accuracy(**metric_param)
        self.valid_auc = AUROC(**metric_param, average="none")
        self.valid_matrix = ConfusionMatrix(**metric_param)
        # -----------------------------
        self.test_acc = Accuracy(**metric_param)
        self.test_auc = AUROC(**metric_param)
        self.test_f1 = F1Score(**metric_param)
        self.test_recall = Recall(**metric_param)
        self.test_prec = Precision(**metric_param)
        self.test_matrix = ConfusionMatrix(**metric_param)

    def forward(self, X):
        return self.module(X)

    def predict_step(self, batch, batch_idx) -> Any:
        return torch.sigmoid(self(batch[0]))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.module(x)
        loss = self.loss_func(y_hat, y, **self.loss_func_hparams)
        y = y.int()
        self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        loss = self.loss_func(y_hat, y, **self.loss_func_hparams)
        y = y.int()
        self.valid_acc(y_hat, y)
        self.valid_auc(y_hat, y)
        self.valid_matrix(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_acc)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.task == "multilabel":
            aucs = self.valid_auc.compute()
            mccs = get_MCCs(self.valid_matrix.compute())
            tensorboard_logger = self.logger.experiment
            # tensorboard_logger.add_scalars(
            #     "mccs", dict([(f"mcc{idx}", imcc) for idx, imcc in enumerate(mccs)]), self.global_step
            # )
            tensorboard_logger.add_scalars(
                "mccs", dict([(f"mcc{idx}", imcc) for idx, imcc in enumerate(mccs)]), self.global_step
            )
            tensorboard_logger.add_scalars(
                "aucs", dict([(f"auc{idx}", iauc) for idx, iauc in enumerate(aucs)]), self.global_step
            )
            Ss = (aucs.cpu() + mccs) * 0.5
            for idx, iS in enumerate(Ss):
                self.log(f"Ss/S{idx}", iS, logger=False)

            auc = aucs.mean()
            mcc = mccs.mean()
        elif self.hparams.task == "binary":
            TP, FN, FP, TN = self.valid_matrix.compute()[[1, 1, 0, 0], [1, 0, 1, 0]]
            numerator = (TP * TN) - (FP * FN)
            denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            if denominator > 0:
                mcc = numerator / (denominator**0.5)
            else:
                mcc = 0
            auc = self.valid_auc.compute()
        self.valid_auc.reset()
        self.valid_matrix.reset()
        S = 0.5 * auc + 0.5 * mcc
        self.log("valid_acc", self.valid_acc)
        self.log("valid_auc", auc, prog_bar=True)
        self.log("valid_mcc", mcc, prog_bar=True)
        self.log("valid_S", S)

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch
        y_hat = self.module(x)
        y = y.int()
        self.test_acc(y_hat, y)
        self.test_auc(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_prec(y_hat, y)
        self.test_matrix(y_hat, y)

    def on_test_epoch_end(self) -> None:
        if self.hparams.task == "binary":
            TP, FN, FP, TN = self.test_matrix.compute()[[1, 1, 0, 0], [1, 0, 1, 0]]
            # 计算敏感性Sn
            Sn = TP / (TP + FN + 1e-06)
            # 计算特异性Sp
            Sp = TN / (FP + TN + 1e-06)
            self.log("Sn", Sn)
            self.log("Sp", Sp)
            numerator = (TP * TN) - (FP * FN)
            denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            if denominator > 0:
                mcc = numerator / (denominator**0.5)
            else:
                mcc = 0
        elif self.hparams.task == "multilabel":
            mccs = get_MCCs(self.test_matrix.compute())
            # tensorboard_logger = self.logger.experiment
            # tensorboard_logger.add_scalars(
            #     "mccs", dict([(f"mcc{idx}", imcc) for idx, imcc in enumerate(mccs)]), self.global_step
            # )
            mcc = mccs.mean()
        self.log("Acc", self.test_acc)
        self.log("AUC", self.test_auc)
        self.log("MCC", mcc)
        self.test_matrix.reset()
        self.log("F1", self.test_f1)
        self.log("Recall", self.test_recall)
        self.log("Precision", self.test_prec)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay  # type: ignore
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="max", factor=0.6, patience=7, min_lr=1e-5
        )
        config = {
            "optimizer": optimizer,
            # "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "valid_S", "frequency": 1},
        }
        return config
