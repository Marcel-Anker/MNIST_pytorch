from CNNConfig import CNNConfig
from app.Data import Data
from CNNModel import CNNModel
from app.Trainer import Trainer

if __name__ == "__main__":
    config = CNNConfig(batchsize=16, lr=0.002, number_conv_layers=2, kernel_size=3, out_channels=4, conv_stride=1)

    data_module = Data(config)
    train_loader, val_loader, test_loader = data_module.get_loaders()

    modelBNorm = CNNModel(config)

    trainerBNorm = Trainer(modelBNorm, config)

    trainerBNorm.train(train_loader, val_loader)
    test_acc_bnorm = trainerBNorm.evaluate(test_loader)

    print(f"Final BNorm Test Accuracy: {test_acc_bnorm:.2f}%")