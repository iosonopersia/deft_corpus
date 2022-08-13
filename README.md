# Definition detector

## Setup

Install the required dependencies:

```bash
pip install -r ./requirements.txt
```

Download the original DEFT Corpus inside the project folder:

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

## Acknowledgments

The [original dataset](https://github.com/adobe-research/deft_corpus) is attributed to [AdobeResearch](https://github.com/adobe-research).
