import os
import shutil
from datetime import datetime
from MLPConfig import MLPConfig
from app.Runner import Runner
from app.Metrics import Metrics

if os.path.exists("app/MLP/runs"): # clearing runs directory so that tensorboard shows recent statistics (tensorboard might still show cached data)
    shutil.rmtree("app/MLP/runs")

os.remove("/Users/marcelanker/PycharmProjects/MINST_pytorch/app/MLP/best_model.pt") # removing best model just in case

best_metric: Metrics | None = None
finalLr = [0.0005, 0.001, 0.002, 0.005, 0.01]
finalBatchsize = [4, 8, 16, 32, 64]
finalHiddenLayerSize = [32, 64, 128, 256, 512]
finalNumberOfHiddenLayers = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    for lr in finalLr:
        for batchsize in finalBatchsize:
            for hidden_layer_size in finalHiddenLayerSize:
                for number_of_hidden_layers in finalNumberOfHiddenLayers:
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

print(f"Final best validation accuracy: {best_metric.final_best_val:.2f} | "
      f"Final best test accuracy: {best_metric.final_best_test} | "
      f"Final epoch mean time: {best_metric.final_epoch_mean_time:.2f}")

best_metric.model.printParameters()

best_metric.drawGraph(f"BestValMLP{datetime.now()}")
best_metric.drawImages(f"WrongImagesOfBestConfMLP{datetime.now()}")