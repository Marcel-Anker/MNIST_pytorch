from typing import Union

import torch.nn
from torch import Tensor
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
            best_val_acc = 0.0
            running_train_loss = 0.0
            running_correct = 0
            total_samples = 0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                running_correct += (preds == labels).float().sum().item()
                total_samples += labels.size(0)

            train_acc = 100 * running_correct / total_samples
            val_acc, running_val_loss, wrong_images = self.evaluate(val_loader)
            train_loss = running_train_loss / len(train_loader)
            val_loss = running_val_loss / len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                counter += 1

            if self.config.__module__ == MLPConfig.__name__:  # ich hasse isinstance :/
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | Validation Loss: {val_loss:.4f} | Training Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Training Acc: {train_acc:.2f}% | "
                    f"Learning Rate: {self.config.learning_rate} | Batch Size: {self.config.batchsize} | "
                    f"Number of Hidden Layers: {self.config.number_of_hidden_layers} | Hidden Layer Size: {self.config.hidden_layer_size}")

            if self.config.__module__ == CNNConfig.__name__:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | Validation Loss: {val_loss:.4f} | Training Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Training Acc: {train_acc:.2f}% | "
                    f"Learning Rate: {self.config.learning_rate} | Batch Size: {self.config.batchsize} | "
                    f"Number of Conv and Pool Layers: {self.config.number_conv_layers} | Kernel Size: {self.config.kernel_size} | "
                    f"Out Channels: {self.config.out_channels}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.config.__module__ == MLPConfig.__name__:
                    model = MLP(
                        lr=self.config.learning_rate,
                        batchsize=self.config.batchsize,
                        hidden_layer_size=self.config.hidden_layer_size,
                        number_of_hidden_layers=self.config.number_of_hidden_layers,
                    )
                    trainMetrics.model = valMetrics.model = model

                if self.config.__module__ == CNNConfig.__name__:
                    model = CNN(
                        lr=self.config.learning_rate,
                        batchsize=self.config.batchsize,
                        number_of_kernels=self.config.number_conv_layers,
                        kernel_size=self.config.kernel_size,
                        stride=self.config.conv_stride,
                    )
                    trainMetrics.model = valMetrics.model = model

            trainMetric = TrainingMetric(loss=train_loss, epoch=epoch, acc=train_acc)
            valMetric = TrainingMetric(loss=val_loss, epoch=epoch, acc= val_acc, wrong_val_images=wrong_images)

            trainMetrics.appendMetric(trainMetric)
            valMetrics.appendMetric(valMetric)

            if counter >= self.config.patience:
                print("early stopped")
                return valMetrics, trainMetrics

        return valMetrics, trainMetrics


    def evaluate(self, loader) -> tuple[float, int, list]:
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0
        wrong_images = []
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                wrong_mask: Tensor = predicted != labels
                if wrong_mask.any():
                    wrong_images.append(inputs[wrong_mask])
                correct += torch.eq(predicted,labels).sum().item()
                total += labels.size(0)
                acc = 100 * correct / total
                running_loss += loss.item()
        return acc, running_loss, wrong_images
