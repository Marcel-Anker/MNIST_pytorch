from typing import List
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class TrainingMetric(BaseModel):
    loss: float
    epoch: int
    acc: float
    wrong_val_images: List[Tensor] = []
    epoch_time: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)