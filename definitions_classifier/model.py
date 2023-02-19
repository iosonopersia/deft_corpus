from transformers import RobertaModel, logging

logging.set_verbosity_error()

import torch
from torch.nn import Sequential


class RoBERTaSentenceClassifier(torch.nn.Module):
  """ See: https://github.com/avramandrei/UPB-SemEval-2020-Task-6/blob/77d92e9c386f270af6ed1db259d3ba6e8bde307b/task1/model.py#L49-L80 """

  def __init__(self, vocab_size: int, backbone: str, cls_head_hidden_units: int):
    super(RoBERTaSentenceClassifier, self).__init__()

    if backbone not in ['roberta-base', 'roberta-large']:
        raise ValueError(f"Invalid backbone: {backbone}")

    self.roberta_model = RobertaModel.from_pretrained(backbone, add_pooling_layer=False)
    self.roberta_model.resize_token_embeddings(vocab_size)
    embedding_size = self.roberta_model.config.hidden_size

    self.classification_head = Sequential(
        # First layer
        torch.nn.Linear(embedding_size, cls_head_hidden_units),
        torch.nn.Dropout(0.8),
        torch.nn.GELU(),

        # Second layer
        torch.nn.Linear(cls_head_hidden_units, cls_head_hidden_units),
        torch.nn.Dropout(0.8),
        torch.nn.GELU(),

        # Output layer
        torch.nn.Linear(cls_head_hidden_units, 1)
    )

  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    roberta_output = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = roberta_output.last_hidden_state
    cls_embeddings = embeddings[:, 0, :]

    output = self.classification_head(cls_embeddings)
    return output
