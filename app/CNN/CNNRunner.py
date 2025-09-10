from CNNConfig import CNNConfig
from app.Data import Data
from CNNBNormModel import CNNBNormModel
from app.Trainer import Trainer


class CNNRunner():
    def __init__(self, config):
        self.config = config

    def startModel(self ):

        data_module = Data(self.config)
        train_loader, val_loader, test_loader = data_module.get_loaders()

        modelBNorm = CNNBNormModel(self.config)

        trainer = Trainer(modelBNorm, self.config)

        valMetrics, trainMetrics = trainer.train(train_loader, val_loader)

        trainMetrics.drawGraph()
        valMetrics.drawGraph()

        valMetrics.model.print()

        return trainer, test_loader