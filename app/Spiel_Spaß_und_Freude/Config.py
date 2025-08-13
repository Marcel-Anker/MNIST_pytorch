
class Config:
    def __init__(self, batchsize, lr, epochs, data_root, hidden_layer_size, number_of_hidden_layers):
        self.batchsize = batchsize
        self.learning_rate = lr
        self.epochs = epochs
        self.data_root = data_root
        self.hidden_layer_size = hidden_layer_size
        self.number_of_hidden_layers = number_of_hidden_layers