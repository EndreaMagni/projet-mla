import torch

class Loss(torch.nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
        self.compute_loss = None # Mettre la fonction pour la loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.compute_loss(input, target)
        return loss
