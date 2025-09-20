


class MLPConfig:

    def __init__(self, batchsize, lr, hidden_layer_size, number_of_hidden_layers, epochs = 40, patience = 7):
        self.batchsize = batchsize
        self.learning_rate = lr
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
        self.number_of_hidden_layers = number_of_hidden_layers - 1
        self.patience = patience

    def printParameters(self):
        print(
            f"These scores were achieved with the following parameters: "
            f"Learning Rate: {self.learning_rate} | Batchsize: {self.batchsize} | Hidden Layer Size: {self.hidden_layer_size} | "
            f"Number of Hidden Layers: {self.number_of_hidden_layers + 1}")