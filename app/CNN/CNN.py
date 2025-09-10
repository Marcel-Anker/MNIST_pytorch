from pydantic import BaseModel


class CNN(BaseModel):
    lr: float
    batchsize: int
    number_of_kernels: int
    kernel_size: int
    stride: int

    def print(self) -> None:
        print(
            f"Learning Rate: {self.lr} | Batchsize: {self.batchsize} | Number of Conv Layers: {self.number_of_kernels} | "
            f"Kernel size: {self.kernel_size} | Stride: {self.stride}"
        )