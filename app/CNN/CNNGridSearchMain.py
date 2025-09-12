from CNNConfig import CNNConfig
from CNNRunner import CNNRunner
from app.Metrics import Metrics

best_metric: Metrics | None = None
if __name__ == "__main__":
    for lr in [0.001, 0.002]:
        for batchsize in [16]:
            for number_conv_layers in [2]:
                for kernel_size in [3]:
                    for conv_stride in [1]:
                        config = CNNConfig(batchsize=batchsize, lr=lr, number_conv_layers=number_conv_layers, kernel_size=kernel_size, out_channels=4, conv_stride=conv_stride, epochs=40, patience=7)

                        runner = CNNRunner(config)

                        trainer, test_loader, valMetrics = runner.startModel()

                        test_acc, _ = trainer.evaluate(test_loader)

                        print(f"Final Test Accuracy: {test_acc:.2f}%")

                        valMetrics.final_best_val = valMetrics.getElementByIndex(config.patience).acc
                        valMetrics.final_best_test = test_acc

                        if best_metric == None:
                            best_metric = valMetrics

                        best_metric = valMetrics.checkBestMetric(best_metric=best_metric, config=config)



print(f"Final best validation accuracy: {best_metric.final_best_val} | "
      f"Final best test accuracy: {best_metric.final_best_test} | "
      f"With the following Hyperparameters: {best_metric.model.print()}")