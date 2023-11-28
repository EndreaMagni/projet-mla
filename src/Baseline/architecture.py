import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class Translator(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return output
