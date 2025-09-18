from time import process_time_ns

from pydantic import BaseModel


class MLP(BaseModel):
    lr: float
    batchsize: int
    hidden_layer_size: int
    number_of_hidden_layers: int

    def printParameters(self):
        print(
            f"These scores were achieved with the following parameters: "
            f"Learning Rate: {self.lr} | Batchsize: {self.batchsize} | Hidden Layer Size: {self.hidden_layer_size} | "
            f"Number of Hidden Layers: {self.number_of_hidden_layers + 1}")