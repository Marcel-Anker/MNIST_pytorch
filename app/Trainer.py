from typing import Union

import torch.nn
from app.MLP.MLPConfig import MLPConfig
from app.CNN.CNNConfig import CNNConfig
from app.CNN.CNNBNormModel import CNNBNormModel
from app.MLP.MLPModel import MLPModel
from app.TrainingMetric import TrainingMetric
from app.Metrics import Metrics
from app.CNN.CNN import CNN
from app.MLP.MLP import MLP


class Trainer:
    def __init__(self, model, config):
        self.model: Union[MLPModel, CNNBNormModel] = model
        self.config: Union[MLPConfig, CNNConfig] = config
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)

    def train(self, train_loader, val_loader) -> tuple[Metrics, Metrics]:

            valMetrics, trainMetrics = self.runEpochs(train_loader, val_loader)

            return valMetrics, trainMetrics

    def runEpochs(self, train_loader, val_loader) -> tuple[Metrics, Metrics]:
        best_val_loss = 2.0 #hoher Wert damit early Stopping ansetzt
        counter = 0
        trainMetrics = Metrics()
        valMetrics = Metrics()
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            total_samples = 0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                running_correct += (preds == labels).float().sum().item()
                total_samples += labels.size(0)

            train_acc = 100 * running_correct / total_samples
            val_acc = self.evaluate(val_loader)

            if self.config.__module__ == MLPConfig.__name__: # ich hasse isinstance :/
                print(f"Epoch {epoch + 1}/{self.config.epochs} | Loss: {running_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Training Acc: {train_acc:.2f}% | "
                    f"Number of Hidden Layers: {self.config.number_of_hidden_layers} | Hidden Layer Size: {self.config.hidden_layer_size} | "
                    f"Learning Rate: {self.config.learning_rate} | Batch Size: {self.config.batchsize}")
            if self.config.__module__ == CNNConfig.__name__:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | Loss: {running_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Training Acc: {train_acc:.2f}% | "
                    f"Learning Rate: {self.config.learning_rate} | Batch Size: {self.config.batchsize}")

    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += torch.eq(predicted,labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total
