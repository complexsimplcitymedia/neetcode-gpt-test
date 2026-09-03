import torch
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        return torch.reshape(to_reshape, (-1, 2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        return torch.mean(to_avg, dim=0)

    def concatenate(self, cat_one, cat_two) -> TensorType[float]:
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(self, prediction, target) -> TensorType[float]:
        return F.mse_loss(prediction, target)