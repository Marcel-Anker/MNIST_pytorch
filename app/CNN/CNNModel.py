import torch.nn as nn
import torch.nn.functional as functions

class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, kernel_size=self.config.kernel_size, padding=self.config.padding)
        for i in range(self.config.number_of_hidden_layers):
            setattr(self, f"fc{i}", nn.Linear(self.config.hidden_layer_size, self.config.hidden_layer_size))
        self.lastFc= nn.Linear(self.config.hidden_layer_size, 10) #0-9 als prediction

    def forward(self, result):
        result = result.view(result.size(0), -1)
        result = functions.relu(self.firstFc(result))
        for i in range(self.config.number_of_hidden_layers):
            result = getattr(self, f"fc{i}")(result)
        result = self.lastFc(result)
        return result