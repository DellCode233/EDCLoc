from utils.LitAuto import LitAuto
from torch.utils.data import DataLoader, Dataset
import torch
import lightning.pytorch as pl
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, MLFlowLogger
from typing import Literal
from typing import Optional, Union
import os


class LitModel(object):
    def __init__(self, estimator, hparams: dict, model_params: dict) -> None:
        """
        `batchsize`\n
        `lr`: learning rate\n
        `name` : set model name\n
        `task` : binary, multilabel, multiclass
            num_labels, num_classes
        `early_stopping`:
            patience : early stopping patience, default 30
            monitor : default valid_S (0.5 * MCC + 0.5 * AUC)
            mode : monitor mode, min or max
        `loss_func`: Callable\n
        `ckpts`:
            monitor : valid metric name
            mode : monitor mode, min or max
            save_last: bool | None = None,
            save_top_k: int = 1,
        `max_epochs` : default 400\n
        `seed` : default 1024\n
        `clip_value` : 0.5 - 2\n
        `clip_algo` : norm or value\n
        """
        self.model_params = model_params
        self.hparams = hparams
        self.estimator = estimator
        self.update_params()

    def load_ckpt(self, ckpt_path):
        if not hasattr(self, "trainer"):
            self.refresh(False)
        self.net = self.__model_type.load_from_checkpoint(
            ckpt_path, module=self.estimator, model_params=self.model_params, **self.hparams
        )

    def update_params(self, model_params: dict = {}, hparams: dict = {}):
        self.model_params.update(model_params)
        self.hparams.update(hparams)

        self.__name = self.hparams["name"]
        self.__batchsize = self.hparams["batchsize"]
        self.__seed = self.hparams.get("seed", 1024)
        self.__max_epochs = self.hparams.get("max_epochs", 400)
        self.__clip_value = self.hparams.get("clip_value", None)
        self.__clip_algo = self.hparams.get("clip_algo", None)
        # -------------------- early stopping --------------
        self.__early_stopping = self.hparams.get("early_stopping", dict(patience=30, monitor="valid_S", mode="max"))
        # -------------------- ckpt ------------------------
        self.__ckpts = self.hparams.get(
            "ckpts", [dict(mode=self.__early_stopping["mode"], monitor=self.__early_stopping["monitor"])]
        )
        # -----------------------------------------------------
        self.__enable_model_summary = self.hparams.get("model_summary", False)
        self.__enable_progress_bar = self.hparams.get("progress_bar", False)
        self.__logdir = self.hparams.get("logdir", None)
        self.__logname = self.hparams.get("logname", self.__name)
        self.__logtype = self.hparams.get("logtype", "TensorBoard")
        self.__model_type = self.hparams.get("model_type", LitAuto)

    def register_logger(self, type_logger: Literal["TensorBoard", "Wandb", "MLFlow"]):
        if type_logger == "TensorBoard":
            return TensorBoardLogger(
                save_dir=self.__logdir if self.__logdir else ".\\tensorboard",
                name=self.__logname if self.__logname else "lightning_logs",
            )
        elif type_logger == "Wandb":
            return WandbLogger(name=self.__logname, save_dir=self.__logdir if self.__logdir else ".\\wandb")
        elif type_logger == "MLFlow":
            return MLFlowLogger(
                experiment_name=self.__logname, save_dir=self.__logdir if self.__logdir else ".\\mlruns"
            )
        else:
            return None

    def refresh(self, create_net=True):
        early_stoping = EarlyStopping(**self.__early_stopping)
        ckpts = [
            ModelCheckpoint(dirpath=os.path.join("ckpts", self.__name), filename=ckpt["monitor"], **ckpt)
            for ckpt in self.__ckpts
        ]

        pl.seed_everything(self.__seed, workers=True)
        if create_net:
            self.net = self.__model_type(self.estimator, self.model_params, **self.hparams)
        self.trainer = L.Trainer(
            gradient_clip_val=self.__clip_value,
            gradient_clip_algorithm=self.__clip_algo,
            logger=self.register_logger(self.__logtype),
            accelerator="gpu",
            max_epochs=self.__max_epochs,
            callbacks=[early_stoping, *ckpts],
            enable_progress_bar=self.__enable_progress_bar,
            enable_model_summary=self.__enable_model_summary,
            # deterministic="warn",
        )

    def fit(self, train_data: Union[Dataset, DataLoader], valid_data: Optional[Union[Dataset, DataLoader]] = None):
        self.train_set, self.valid_set = train_data, valid_data
        self.refresh()
        if isinstance(self.train_set, DataLoader):
            train_iter = self.train_set
        else:
            train_iter = DataLoader(
                self.train_set, self.__batchsize if self.__batchsize > 0 else len(self.train_set), shuffle=True  # type: ignore
            )
        if self.valid_set:
            if isinstance(self.valid_set, DataLoader):
                valid_iter = self.valid_set
            else:
                valid_iter = DataLoader(
                    self.valid_set, self.__batchsize if self.__batchsize > 0 else len(self.valid_set), shuffle=False  # type: ignore
                )
        else:
            valid_iter = None
        self.trainer.fit(self.net, train_dataloaders=train_iter, val_dataloaders=valid_iter)

    def predict_proba(self, dataset: Union[Dataset, DataLoader], ckpt_path="best"):
        if isinstance(dataset, DataLoader):
            data_loader = dataset
        else:
            data_loader = DataLoader(dataset, self.__batchsize if self.__batchsize > 0 else len(dataset), shuffle=False)  # type: ignore
        if not hasattr(self, "trainer"):
            self.refresh()
        return torch.concat(self.trainer.predict(self.net, dataloaders=data_loader, ckpt_path=ckpt_path), dim=0)  # type: ignore

    def test(self, test_data: Union[Dataset, DataLoader], verbose=False, ckpt_path="best"):
        if isinstance(test_data, DataLoader):
            test_loader = test_data
        else:
            test_loader = DataLoader(
                test_data, self.__batchsize if self.__batchsize > 0 else len(test_data), shuffle=False  # type: ignore
            )
        if not hasattr(self, "trainer"):
            self.refresh()
        return self.trainer.test(self.net, dataloaders=test_loader, ckpt_path=ckpt_path, verbose=verbose)[0]

    @property
    def model_ckpt_callbacks(self):
        return self.trainer.checkpoint_callbacks

    @property
    def best_ckpts_path(self):
        return [ckpt_callback.best_model_path for ckpt_callback in self.model_ckpt_callbacks]
