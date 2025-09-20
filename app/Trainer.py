from typing import Union

import time
import torch.nn
from torch import Tensor
from app.MLP.MLPConfig import MLPConfig
from app.CNN.CNNConfig import CNNConfig
from app.CNN.CNNModel import CNNModel
from app.MLP.MLPModel import MLPModel
from app.TrainingMetric import TrainingMetric
from app.Metrics import Metrics


class Trainer:
    def __init__(self, model, config):
        self.model: Union[MLPModel, CNNModel] = model
        self.config: Union[MLPConfig, CNNConfig] = config
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.0)

    def train(self, train_loader, val_loader) -> tuple[Metrics, Metrics]:

            valMetrics, trainMetrics = self.runEpochs(train_loader, val_loader)

            return valMetrics, trainMetrics

    def runEpochs(self, train_loader, val_loader) -> tuple[Metrics, Metrics]:
        best_val_loss = 5.0 #relativly high value so that early stopping works
        patience_counter = 0 #paitence counter for early stopping
        trainMetrics = Metrics()
        valMetrics = Metrics()
        for epoch in range(self.config.epochs):
            start_epoch_time = time.time()
            self.model.train()
            running_train_loss = 0.0
            running_correct = 0
            total_samples = 0
            for inputs, labels in train_loader:
                inputs = inputs.to('mps')
                labels = labels.to('mps')
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
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1

            end_epoch_time = time.time()

            if self.config.__module__ == MLPConfig.__name__:  # isinstance doesnt work :/
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | Validation Loss: {val_loss:.4f} | Training Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Training Acc: {train_acc:.2f}% | "
                    f"Learning Rate: {self.config.learning_rate} | Batch Size: {self.config.batchsize} | "
                    f"Number of Hidden Layers: {self.config.number_of_hidden_layers + 1} | Hidden Layer Size: {self.config.hidden_layer_size} | "
                    f"Elapsed time for epoch: {end_epoch_time-start_epoch_time}")

            if self.config.__module__ == CNNConfig.__name__:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | Validation Loss: {val_loss:.4f} | Training Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Training Acc: {train_acc:.2f}% | "
                    f"Learning Rate: {self.config.learning_rate} | Batch Size: {self.config.batchsize} | "
                    f"Number of Conv and Pool Layers: {self.config.number_conv_layers + 1} | Kernel Size: {self.config.kernel_size} | "
                    f"Out Channels: {self.config.out_channels} | "
                    f"Elapsed time for epoch: {end_epoch_time-start_epoch_time}")


            trainMetric = TrainingMetric(loss=train_loss, epoch=epoch, acc=train_acc, epoch_time=end_epoch_time - start_epoch_time)
            valMetric = TrainingMetric(loss=val_loss, epoch=epoch, acc= val_acc, wrong_val_images=wrong_images)

            trainMetrics.appendMetric(trainMetric)
            valMetrics.appendMetric(valMetric)

            if patience_counter >= self.config.patience:
                print("early stopped")
                valMetrics.early_stopped = trainMetrics.early_stopped = True
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
                inputs = inputs.to('mps')
                labels = labels.to('mps')
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
