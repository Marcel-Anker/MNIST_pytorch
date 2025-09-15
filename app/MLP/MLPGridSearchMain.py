import os
import shutil
from datetime import datetime
from MLPConfig import MLPConfig
from app.Runner import Runner
from app.Metrics import Metrics

if os.path.exists("app/MLP/runs"):
    shutil.rmtree("app/MLP/runs")

best_metric: Metrics | None = None
finalLr = [0.0005, 0.001, 0.002, 0.005, 0.01]
finalBatchsize = [4, 8, 16, 32, 64]
finalHiddenLayerSize = [25, 50, 75, 100, 200]
finalNumberOfHiddenLayers = [5, 20, 50, 100, 200]

if __name__ == "__main__":
    for lr in [0.001]:
        for batchsize in [16]:
            for hidden_layer_size in [100]:
                for number_of_hidden_layers in [100]:
                    config = MLPConfig(batchsize=batchsize, lr=lr, hidden_layer_size=hidden_layer_size, number_of_hidden_layers=number_of_hidden_layers, epochs=40,
                                       patience=7)

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

print(f"Final best validation accuracy: {best_metric.final_best_val} | "
      f"Final best test accuracy: {best_metric.final_best_test} | "
      f"With the following Hyperparameters: {best_metric.model.print()}")

best_metric.drawGraph(f"BestValMLP{datetime.now()}")
best_metric.drawImages(f"WrongImagesOfBestConfMLP{datetime.now()}")