import torch
import torchvision
from pydantic import BaseModel, ConfigDict
from typing import List
from torch import Tensor
from app.TrainingMetric import TrainingMetric
from torch.utils.tensorboard import SummaryWriter
from typing import Self


class Metrics(BaseModel):

    metrics: List[TrainingMetric] = []
    final_best_val: float = 0.0
    final_best_test: float = 0.0
    wrong_test_images: List[Tensor] = []
    final_epoch_mean_time: float = 0.0
    early_stopped: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def appendMetric(self, metric) -> None:
        self.metrics.append(metric)

    def getBestMetricElement(self, patience) -> TrainingMetric | None:
        if not self.metrics:
            print("Metrics empty")

        if self.early_stopped:
            return self.metrics[patience]
        else:
            return self.metrics[-1]

    def drawGraph(self, type: str) -> None:

        writer = SummaryWriter(f"runs/mnist/{type}")
        for metric in self.metrics:
            writer.add_scalar(f"Loss Function {type}", metric.loss, global_step=metric.epoch)

        writer.close()

    def drawImages(self, type: str) -> None:

        writer = SummaryWriter(f"runs/mnist{type}")

        wrong_images = torch.cat(self.wrong_test_images)

        img_grid = torchvision.utils.make_grid(wrong_images[:36], nrow=6, normalize=True) #16, da Stichprobe reicht

        writer.add_image(f"Falsch klassifizierte Bilder von {type}", img_grid)
        writer.close()

    def checkBestMetric(self, best_metric: Self, config) -> Self:
        if self.getBestMetricElement(
                (config.patience * -1) - 1).loss < best_metric.getBestMetricElement(
            (config.patience * -1) - 1).loss:
            return self
        else:
            return best_metric

    def getMeanEpochTime(self) -> float:
        mean_time = 0.0
        len = 0
        for metric in self.metrics:
            mean_time += metric.epoch_time
            len += 1
        return mean_time / len