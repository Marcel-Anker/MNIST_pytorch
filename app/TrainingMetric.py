from typing import List
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class TrainingMetric(BaseModel):
    loss: float
    epoch: int
    acc: float
    wrong_val_images: List[Tensor] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)