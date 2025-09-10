from pydantic import BaseModel


class TrainingMetric(BaseModel):
    loss: float
    epoch: int

