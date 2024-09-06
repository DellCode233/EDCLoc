from utils.prepare_data import one_hot, iDataset  # , dirnaphyche
import pandas as pd
from typing import Literal


def load_datasets(ifold: int):
    trainset = pd.read_csv(f"5_fold_data/Train_fold{ifold}.csv", index_col=0)
    validset = pd.read_csv(f"5_fold_data/Validation_fold{ifold}.csv", index_col=0)
    testset = pd.read_csv(f"5_fold_data/Test_fold{ifold}.csv", index_col=0)
    return (
        iDataset(
            [
                trainset["seq"],
            ],
            trainset.loc[:, ["Nucleus", "Exosome", "Cytosol", "Ribosome", "Membrane", "Endoplasmic reticulum"]],
            fe=one_hot,
        ),
        iDataset(
            [
                validset["seq"],
            ],
            validset.loc[:, ["Nucleus", "Exosome", "Cytosol", "Ribosome", "Membrane", "Endoplasmic reticulum"]],
            fe=one_hot,
        ),
        iDataset(
            [
                testset["seq"],
            ],
            testset.loc[:, ["Nucleus", "Exosome", "Cytosol", "Ribosome", "Membrane", "Endoplasmic reticulum"]],
            fe=one_hot,
        ),
    )
