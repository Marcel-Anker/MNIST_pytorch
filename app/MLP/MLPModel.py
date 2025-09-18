import torch.nn as nn
import torch.nn.functional as functions

class MLPModel(nn.Module):
    def __init__(self, config):
        super(MLPModel, self).__init__()
        self.config = config
        self.firstFc = nn.Linear(in_features=28 * 28, out_features=self.config.hidden_layer_size) #28 * 28 mnist picture
        for i in range(self.config.number_of_hidden_layers):
            setattr(self, f"fc{i}", nn.Linear(in_features=self.config.hidden_layer_size, out_features=self.config.hidden_layer_size))
            setattr(self, f"bnorm{i}", nn.BatchNorm2d(self.config.hidden_layer_size))
        self.lastFc= nn.Linear(in_features=self.config.hidden_layer_size, out_features=10) #0-9 als prediction
        self.to('mps')

    def forward(self, result):
        result = result.view(result.size(0), -1)
        result = functions.relu(self.firstFc(result))
        for i in range(self.config.number_of_hidden_layers):
            result = functions.relu(getattr(self, f"fc{i}")(result))
        result = self.lastFc(result)
        return result