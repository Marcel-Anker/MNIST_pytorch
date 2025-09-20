

class CNNConfig():

    def __init__(self, batchsize, lr, number_conv_layers, kernel_size, conv_stride, epochs=40, patience=7, out_channels = 4):
        self.batchsize = batchsize
        self.learning_rate = lr
        self.epochs = epochs
        self.number_conv_layers = number_conv_layers - 1
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv_stride = conv_stride
        self.patience = patience

    def printParameters(self):
        print(
            f"These scores were achieved with the following parameters: "
            f"Learning Rate: {self.learning_rate} | Batchsize: {self.batchsize} | Number of Conv Layers: {self.number_conv_layers + 1} | "
            f"Kernel size: {self.kernel_size} | Stride: {self.conv_stride}"
        )