import torch
from model import RoBERTaSentenceClassifier
from tools.checkpoint_handler import CheckpointHandler
from transformers import RobertaTokenizer, logging
from utils import get_config

logging.set_verbosity_error()


def run_inference(sentence: str, threshold: float = 0.5) -> None:
    model.eval()
    with torch.inference_mode():
        # Tokenize sentence
        tokens = tokenizer([sentence], add_special_tokens=False, max_length=MAX_LENGTH,
                        padding="longest", truncation="longest_first",
                        return_attention_mask=True, return_tensors="pt")
        sentences = tokens["input_ids"].to(DEVICE)
        masks = tokens["attention_mask"].to(DEVICE)

        # Forward pass
        predictions = model(input_ids=sentences, attention_mask=masks)

        # Logits to probabilities
        probability = torch.sigmoid(predictions).item()
        print(f"p(IS_DEF=True|input, θ) = {probability*100:.2f}%", end="\t")
        print("[DEFINITION]" if probability > threshold else "[OTHER]")


if __name__ == "__main__":
     #============DEVICE==============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    model_cfg = config.model
    test_cfg = config.test

    # =============TOOLS==============
    checkpoint_handler = CheckpointHandler(config.checkpoint)

    # ===========DATASET==============
    TEST_PATH = dataset_cfg.test_labels_file
    MAX_LENGTH = dataset_cfg.max_sentence_length

    # ========== DATASET=============
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.add_tokens(["<link>", "<equation>"])

    # ===========MODEL===============
    model = RoBERTaSentenceClassifier(len(tokenizer), **model_cfg)
    model = model.to(DEVICE)
    checkpoint_handler.load_for_testing(test_cfg.checkpoint_path, model)

    while True:
        sentence = input(">>> ")
        run_inference(sentence, test_cfg.threshold)
        print()
