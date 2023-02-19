import torch

class CustomBCELoss(torch.nn.Module):
    def __init__(self, class_weights: dict[int, float] = None, label_smoothing: float = 0.0, reduction: str = 'mean'):
        super(CustomBCELoss, self).__init__()

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")

        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = self.bce_loss(input, target)

        if self.class_weights is not None:
            positive_weight = torch.tensor(self.class_weights[1], device=target.device)
            negative_weight = torch.tensor(self.class_weights[0], device=target.device)
            weights = torch.where(target > 0.5, positive_weight, negative_weight)
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean(dim=0)
        elif self.reduction == 'sum':
            return loss.sum(dim=0)
        else:
            return loss
