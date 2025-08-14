class CNNConfig:
    def __init__(self, batchsize, lr, epochs, number_convolution_layer, number_pooling_layer, kernel_size, padding):
        self.batchsize = batchsize
        self.learning_rate = lr
        self.epochs = epochs
        self.number_convolution_layer = number_convolution_layer
        self.number_pooling_layer = number_pooling_layer
        self.kernel_size = kernel_size
        self.padding = padding