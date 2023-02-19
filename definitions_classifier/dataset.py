import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DefinitionsFactsDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_sequence_length: int = 60):
        df = pd.read_csv(dataset_path, sep="\t", header=0, encoding='utf-8',
                         names=["SENTENCE", "HAS_DEF"], usecols=["SENTENCE", "HAS_DEF"],
                         dtype={"SENTENCE": str, "HAS_DEF": np.uint8})

        X, y = df["SENTENCE"].to_list(), df["HAS_DEF"].to_numpy()

        encodings = tokenizer(X, add_special_tokens=True, max_length=max_sequence_length,
                                padding="longest", truncation="longest_first",
                                return_attention_mask=True, return_tensors="pt")

        self.sentences = encodings['input_ids']
        self.labels = torch.from_numpy(y).unsqueeze(dim=-1).float()
        self.masks = encodings['attention_mask']

        self.length = self.labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"sentences": self.sentences[idx], "labels": self.labels[idx], "masks": self.masks[idx]}

    def get_class_weights(self):
        def_samples = torch.sum(self.labels).item()

        non_def_samples = self.length - def_samples

        return {0: def_samples / self.length, 1: non_def_samples / self.length}
