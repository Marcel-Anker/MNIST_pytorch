from pydantic import BaseModel, ConfigDict, SkipValidation
from typing import List
from app.TrainingMetric import TrainingMetric
from app.CNN.CNN import CNN
from app.MLP.MLP import MLP
from torch.utils.tensorboard import SummaryWriter
from typing import Self


class Metrics(BaseModel):

    metrics: List[TrainingMetric] = []
    model: (CNN, MLP) = None
    final_best_val: float = 0.0
    final_best_test: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def appendMetric(self, metric) -> None:
        self.metrics.append(metric)

    def getElementByIndex(self, index: int) -> TrainingMetric | None:
        if not self.metrics:
            print("Metrics empty")

        try:
            return self.metrics[index]
        except IndexError:
            return self.metrics[-1]

    def drawGraph(self, type: str) -> None:

        writer = SummaryWriter(f"runs/mnist/{type}")
        for metric in self.metrics:
            writer.add_scalar(f"Loss Function {type}", metric.loss, global_step=metric.epoch)

        writer.close()

    def checkBestMetric(self, best_metric: Self, config) -> Self:
        if self.getElementByIndex(
                (config.patience * -1) - 1).loss < best_metric.getElementByIndex(
            (config.patience * -1) - 1).loss:
            return self
        else:
            return best_metric