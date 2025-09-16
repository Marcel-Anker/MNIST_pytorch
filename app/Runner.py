from pydantic import BaseModel
from app.Data import Data
from pydantic import BaseModel, ConfigDict
from app.CNN.CNNModel import CNNModel
from app.MLP.MLPModel import MLPModel
from app.MLP.MLPConfig import MLPConfig
from app.CNN.CNNConfig import CNNConfig
from app.Trainer import Trainer
from app.CNN.CNN import CNN


class Runner(BaseModel):
    config: (CNNConfig, MLPConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    def startModel(self ):

        data_module = Data(self.config)
        train_loader, val_loader, test_loader = data_module.get_loaders()

        if self.config.__module__ == CNNConfig.__name__:
            model = CNNModel(self.config)
            #print("cnn")
        else:
            model = MLPModel(self.config)
            #print("mlp")

        trainer = Trainer(model, self.config)

        valMetrics, trainMetrics = trainer.train(train_loader, val_loader)


        if self.config.__module__ == CNNConfig.__name__:
            trainMetrics.drawGraph("CNNTrain")
            valMetrics.drawGraph("CNNVal")
        else:
            trainMetrics.drawGraph("MLPTrain")
            valMetrics.drawGraph("MLPVal")


        return trainer, test_loader, valMetrics