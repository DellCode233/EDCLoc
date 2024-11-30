# **EDCLoc: A Prediction Model for mRNA Subcellular Localization Using Improved Focal Loss to Address Multi-Label Class Imbalance**

**EDCLoc** is a novel multi-label classifier designed to predict the subcellular localization of mRNA, a critical factor in regulating gene expression and cellular function. Traditional experimental methods for mRNA localization are time-consuming and costly, and existing prediction tools face limitations in handling sequence length variations and high-dimensional data. **EDCLoc** addresses these challenges with a more efficient and accurate approach.

## **Requirements**

To use this project, you need to install the following libraries:
- `lightning==2.0.9.post0`
- `multimethod==1.9.1`
- `numpy==1.24.4`
- `pandas==2.0.3`
- `scikit_learn==1.3.2`
- `torch==2.0.0+cu118`
- `torch==2.0.1`
- `torchmetrics==1.2.0`

You can install them by running the following command:

```bash
pip install -r requirements.txt
```

## **Usage**

To use the EDCLoc model, run the following command:

```bash
python EDCLoc_predict.py -h
```

**usage**: `EDCLoc_predict.py` [-h] --input INPUT [--output OUTPUT]

**EDCLoc**: A Prediction Model for mRNA Subcellular Localization

**optional arguments**:
- `-h, --help`       : show this help message and exit
- `--input INPUT`    : Query mRNA sequences in fasta format
- `--output OUTPUT`  : The path where you want to save the prediction results

