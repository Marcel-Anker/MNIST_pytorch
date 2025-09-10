from pydantic import BaseModel

class CNNConfig():
    def __init__(self, batchsize, lr, number_conv_layers, kernel_size, conv_stride, epochs, patience, out_channels = 4):
        self.batchsize = batchsize
        self.learning_rate = lr
        self.epochs = epochs
        self.number_conv_layers = number_conv_layers - 1
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv_stride = conv_stride
        self.patience = patience
