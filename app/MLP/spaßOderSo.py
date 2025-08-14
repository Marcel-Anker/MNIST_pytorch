from MLPConfig import MLPConfig
from app.Data import Data
from MLPModel import MLPModel
from app.Trainer import Trainer

if __name__ == "__main__":
    config = MLPConfig(batchsize=6, lr=0.001, epochs=1, hidden_layer_size=100, number_of_hidden_layers=1)

    data_module = Data(config)
    train_loader, val_loader, test_loader = data_module.get_loaders()

    model = MLPModel(config)
    trainer = Trainer(model, config)

    trainer.train(train_loader, val_loader)
    test_acc = trainer.evaluate(test_loader)
    print(f"Final Test Accuracy: {test_acc:.2f}%")