from Config import Config
from Data import Data
from Model import Model
from Trainer import Trainer

if __name__ == "__main__":
    config = Config(batchsize=32, lr=0.001, epochs=1, data_root="./data", hidden_layer_size=50, number_of_hidden_layers=2)

    data_module = Data(config)
    train_loader, val_loader, test_loader = data_module.get_loaders()

    model = Model(config)
    trainer = Trainer(model, config)

    trainer.train(train_loader, val_loader)
    test_acc = trainer.evaluate(test_loader)
    print(f"Final Test Accuracy: {test_acc:.2f}%")