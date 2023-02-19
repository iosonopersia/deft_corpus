import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from model import RoBERTaSentenceClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)
from tools.checkpoint_handler import CheckpointHandler
from tqdm import tqdm
from transformers import RobertaTokenizer, logging
from utils import get_config

from dataset import DefinitionsFactsDataset

logging.set_verbosity_error()

def test_loop():
    # ========== DATASET=============
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.add_tokens(["<link>", "<equation>"])

    test_data = DefinitionsFactsDataset(TEST_PATH, tokenizer, max_sequence_length=MAX_LENGTH)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_cfg.test)

    # ===========MODEL===============
    model = RoBERTaSentenceClassifier(len(tokenizer), **model_cfg)
    model = model.to(DEVICE)
    checkpoint_handler.load_for_testing(test_cfg.checkpoint_path, model)

    loop = tqdm(test_loader, leave=True)
    loop.set_description(f"Test")

    model_predictions = []
    ground_truth_labels = []

    model.eval()
    with torch.inference_mode():
        for _, batch in enumerate(loop):
            # Get batch data
            sentences = batch['sentences'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            masks = batch['masks'].to(DEVICE)

            # Forward pass
            predictions = model(input_ids=sentences, attention_mask=masks)

            # Logits to probabilities
            predictions = torch.sigmoid(predictions)

            model_predictions.append(predictions)
            ground_truth_labels.append(labels)

    model_predictions = torch.cat(model_predictions, dim=0)
    ground_truth_labels = torch.cat(ground_truth_labels, dim=0)

    pickle.dump(model_predictions, open("model_predictions.pkl", "wb"))
    pickle.dump(ground_truth_labels, open("ground_truth_labels.pkl", "wb"))


def compute_metrics():
    model_predictions = pickle.load(open("model_predictions.pkl", "rb"))
    true_labels = pickle.load(open("ground_truth_labels.pkl", "rb"))

    # From tensors to numpy arrays
    model_predictions = model_predictions.cpu().squeeze(dim=-1).numpy()
    true_labels = true_labels.cpu().squeeze(dim=-1).numpy()

    best_threshold = plot_prec_rec_curve(true_labels, model_predictions)

    predicted_labels = (model_predictions > best_threshold).astype(int)
    true_labels = true_labels.astype(int)

    test_acc = 100*accuracy_score(true_labels, predicted_labels)
    test_rec = 100*recall_score(true_labels, predicted_labels)
    test_prec = 100*precision_score(true_labels, predicted_labels)
    test_f1 = 100*f1_score(true_labels, predicted_labels)

    print("TEST RESULTS:")
    print(f"Accuracy: {round(test_acc, 2)}%")
    print(f"Precision: {round(test_prec, 2)}%")
    print(f"Recall: {round(test_rec, 2)}%")
    print(f"F1 Score: {round(test_f1, 2)}%")

    plot_confusion_matrix(true_labels, predicted_labels)

    # Save results
    save_results_to_tsv(predicted_labels)


def plot_prec_rec_curve(true_labels: np.array, model_predictions: np.array) -> float:
    prec, rec, thresholds = precision_recall_curve(true_labels, model_predictions, pos_label=1)

    # Locate the index of the largest f1 score
    f1_score = 2 * prec * rec / (prec + rec + 1e-8)
    ix = np.argmax(f1_score)

    plt.figure(figsize=(5, 5))

    plt.plot(rec, prec, marker='.', label='Model')
    plt.scatter(rec[ix], prec[ix], marker='o', color='black', zorder=1000, label=f'Best threshold: {thresholds[ix]:.2f}')

    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.show()

    return thresholds[ix].item()


def plot_confusion_matrix(true_labels: np.array, predicted_labels: np.array):
    cm = confusion_matrix(true_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Other", "Definition"])
    disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.show()


def save_results_to_tsv(predicted_labels: np.array):
    #! Here we assume that the order of the predictions is the same
    #! as the order of the sentences in the test set
    #! (i.e. shuffle=False in the dataloader)
    df = pd.read_csv(TEST_PATH, sep="\t", header=0, encoding='utf-8',
                        names=["SENTENCE", "HAS_DEF"], usecols=["SENTENCE", "HAS_DEF"],
                        dtype={"SENTENCE": str, "HAS_DEF": np.uint8})
    df['PREDICTION'] = predicted_labels

    df_false_positives = df[(df['HAS_DEF'] == 0) & (df['PREDICTION'] == 1)]
    df_false_negatives = df[(df['HAS_DEF'] == 1) & (df['PREDICTION'] == 0)]
    df_true_positives = df[(df['HAS_DEF'] == 1) & (df['PREDICTION'] == 1)]
    df_true_negatives = df[(df['HAS_DEF'] == 0) & (df['PREDICTION'] == 0)]

    df_false_positives.to_csv("false_positives.tsv", sep="\t", index=False, encoding='utf-8')
    df_false_negatives.to_csv("false_negatives.tsv", sep="\t", index=False, encoding='utf-8')
    df_true_positives.to_csv("true_positives.tsv", sep="\t", index=False, encoding='utf-8')
    df_true_negatives.to_csv("true_negatives.tsv", sep="\t", index=False, encoding='utf-8')


if __name__ == "__main__":
     #============DEVICE==============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    dataloader_cfg = config.dataloader
    model_cfg = config.model
    test_cfg = config.test

    # =============TOOLS==============
    checkpoint_handler = CheckpointHandler(config.checkpoint)

    # ===========DATASET==============
    TEST_PATH = dataset_cfg.test_labels_file
    MAX_LENGTH = dataset_cfg.max_sentence_length

    test_loop()
    compute_metrics()
