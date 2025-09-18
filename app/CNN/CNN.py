from pydantic import BaseModel


class CNN(BaseModel):
    lr: float
    batchsize: int
    number_of_conv_layers: int
    kernel_size: int
    stride: int

    def printParameters(self):
        print(
            f"These scores were achieved with the following parameters: "
            f"Learning Rate: {self.lr} | Batchsize: {self.batchsize} | Number of Conv Layers: {self.number_of_conv_layers + 1} | "
            f"Kernel size: {self.kernel_size} | Stride: {self.stride}"
        )