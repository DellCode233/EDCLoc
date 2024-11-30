import argparse
import os

def sigmoid_focal_loss(
    inputs,
    targets,
    alpha=[0.42, 0.01, 0.76, 0.64, 0.63, 0.66],
    omega=[1.3, 1.0, 0.9, 1.4, 1.2, 1.4],
    gamma: float = 0,
):
    from torch import sigmoid, tensor
    import torch.nn.functional as F

    targets = targets.view_as(inputs).type_as(inputs)
    p = sigmoid(inputs).detach()
    p_t = p * targets + (1 - p) * (1 - targets)
    weights = (1 - p_t) ** gamma
    # target (B, C), alpha (1, C)
    alpha = tensor(alpha, device=inputs.device).type_as(inputs).reshape(1, inputs.shape[1])
    omega = tensor(omega, device=inputs.device).type_as(inputs).reshape(1, inputs.shape[1])
    weights *= alpha * targets + (1 - alpha) * (1 - targets)
    weights *= omega
    loss = F.binary_cross_entropy_with_logits(inputs, targets, weights, reduction="none")
    # loss B x C
    loss = loss.sum() / weights.sum()
    return loss


def load_model():
    from utils.one_trial import LitModel
    from models.EDCLoc import classifier

    model_params = classifier.get_model_params()
    hparams = classifier.get_hparams()
    hparams["loss_func"] = sigmoid_focal_loss
    hparams["loss_func_hparams"] = dict(
        alpha=[0.42, 0.01, 0.76, 0.64, 0.63, 0.66], omega=[1.3, 1.0, 0.9, 1.4, 1.2, 1.4]
    )
    return LitModel(classifier, hparams, model_params)


def record_to_list(record):
    if len(record.seq) > 8000:
        seq = record.seq[:4000] + record.seq[-4000:]
    else:
        seq = record.seq + "N" * (8000 - len(record.seq))
    return seq.transcribe()


def fasta2dataset(fasta_path):
    from utils.prepare_data import iDataset
    from Bio import SeqIO
    from utils.prepare_data import one_hot

    seqs = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seqs.append(record_to_list(record))
        ids.append(record.id)
    return (
        iDataset(
            argc=[
                seqs,
            ],
            fe=one_hot,
        ),
        ids,
    )


def model_predict(input_path, output_path):
    model = load_model()
    import pandas as pd
    import numpy as np

    testset, ids = fasta2dataset(input_path)
    all_y_score = []
    for j in range(6):
        ckpt_path = os.path.join("save_ckpts", f"fold0", f"S{j}.ckpt")
        y_score = model.predict_proba(testset, ckpt_path=ckpt_path)
        all_y_score.append(np.array(y_score[:, j]))
    all_y_score = np.stack(all_y_score, axis=1).round(4)
    all_y_pred = np.where(all_y_score > 0.5, 1, 0)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    pd.DataFrame(
        all_y_score,
        columns=["Nucleus", "Exosome", "Cytosol", "Ribosome", "Membrane", "Endoplasmic reticulum"],
        index=ids,
    ).to_csv(os.path.join(output_path, "score.csv"))
    pd.DataFrame(
        all_y_pred,
        columns=["Nucleus", "Exosome", "Cytosol", "Ribosome", "Membrane", "Endoplasmic reticulum"],
        index=ids,
    ).to_csv(os.path.join(output_path, "pred.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDCLoc: A Prediction Model for mRNA Subcellular Localization")
    parser.add_argument("--input", type=str, required=True, help="Query mRNA sequences in fasta format")
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="The path where you want to save the prediction results",
        default="output_result",
    )
    args = parser.parse_args()

    model_predict(args.input, args.output)
