from CNNConfig import CNNConfig
from CNNRunner import CNNRunner

if __name__ == "__main__":
    for lr in [0.001, 1]:
        for batchsize in [16]:
            for number_conv_layers in [2]:
                for kernel_size in [3]:
                    for conv_stride in [1]:
                        config = CNNConfig(batchsize=batchsize, lr=lr, number_conv_layers=number_conv_layers, kernel_size=kernel_size, out_channels=4, conv_stride=conv_stride, epochs=40, patience=10)

                        runner = CNNRunner(config)

                        trainer, test_loader = runner.startModel()

                        test_acc, _ = trainer.evaluate(test_loader)

                        print(f"Final Test Accuracy: {test_acc:.2f}%")