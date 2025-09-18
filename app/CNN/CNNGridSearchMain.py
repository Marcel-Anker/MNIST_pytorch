import os
import shutil
from datetime import datetime

from sympy.printing.pytorch import torch

from CNNConfig import CNNConfig
from app.Runner import Runner
from app.Metrics import Metrics

best_metric: Metrics | None = None
finalLr = [0.0005, 0.001, 0.002, 0.005, 0.01]
finalBatchsize = [4, 8, 16, 32, 64]
finalNumberConvLayers = [1, 2, 3, 4, 5]
finalKernelSize = [2, 3, 4, 5, 6]
finalConvStride = [1, 2, 3, 4, 5]

if os.path.exists("app/CNN/runs"): # clearing runs directory so that tensorboard shows recent statistics (tensorboard might still show cached data)
    shutil.rmtree("app/CNN/runs")

os.remove(r"../../app/CNN/best_model.pt") # removing best model just in case

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print("MPS device not found.")

if __name__ == "__main__":
    for lr in finalLr:
        for batchsize in finalBatchsize:
            for number_conv_layers in finalNumberConvLayers:
                for kernel_size in finalKernelSize:
                    for conv_stride in finalConvStride:
                        config = CNNConfig(batchsize=batchsize, lr=lr, number_conv_layers=number_conv_layers, kernel_size=kernel_size, out_channels=4, conv_stride=conv_stride, epochs=40, patience=7)

                        runner = Runner(config=config)

                        trainer, test_loader, valMetrics = runner.startModel()

                        test_acc, _, wrong_images = trainer.evaluate(test_loader)

                        print(f"Final Test Accuracy: {test_acc:.2f}%")

                        valMetrics.final_best_val = valMetrics.getElementByIndex(config.patience).acc
                        valMetrics.final_best_test = test_acc
                        valMetrics.wrong_test_images = wrong_images

                        if best_metric == None:
                            best_metric = valMetrics

                        best_metric = valMetrics.checkBestMetric(best_metric=best_metric, config=config)



print(f"Final best validation accuracy: {best_metric.final_best_val:.2f}% | "
      f"Final best test accuracy: {best_metric.final_best_test:.2f}% | "
      f"Final epoch mean time: {best_metric.final_epoch_mean_time:.2f}")

best_metric.model.printParameters()

best_metric.drawGraph(f"BestValCNN{datetime.now()}")
best_metric.drawImages(f"WrongImagesOfBestConfCNN{datetime.now()}")