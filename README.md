# Definition detector

## Setup

Download the original DEFT Corpus:

```bash
git clone https://github.com/adobe-research/deft_corpus
```

Perform the preprocessing step (preprocessed data will be found inside `./dataset/preprocessed/`):

```bash
cd ./dataset
python ./preprocessing.py
```

Perform the augmentation step (preprocessed data will be found inside `./dataset/augmented/`):

```bash
cd ./dataset
python ./augmentation.py
```

Run the Jupyter Notebook (`./TL_SimonePersiani.ipynb`).
