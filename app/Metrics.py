from pydantic import BaseModel, ConfigDict, SkipValidation
from typing import List
from app.TrainingMetric import TrainingMetric
from app.CNN.CNN import CNN
from app.MLP.MLP import MLP
from torch.utils.tensorboard import SummaryWriter


class Metrics(BaseModel):

    metrics: List[TrainingMetric] = []
    model: (CNN, MLP) = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def appendMetric(self, metric) -> None:
        self.metrics.append(metric)

    def getLastValue(self) -> TrainingMetric | None:
        if self.metrics:
            return self.metrics[-1]
        else:
            return print("Metrics empty")


    def drawGraph(self) -> None:
        writer = SummaryWriter("runs/mnist")

        for metric in self.metrics:
            writer.add_scalar("Loss Function Plot", metric.epoch, int(metric.loss))

        writer.close()