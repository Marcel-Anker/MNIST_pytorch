import torch.nn

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)

    def train(self, train_loader, val_loader):
        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch + 1}/{self.config.epochs} | Loss: {running_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Number of Hidden Layers: {self.config.number_of_hidden_layers} | Hidden Layer Size: {self.config.hidden_layer_size} | "
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
