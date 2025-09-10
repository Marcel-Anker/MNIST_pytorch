import torch.nn as nn
import torch.nn.functional as functions

class CNNBNormModel(nn.Module):
    def __init__(self, config):
        super(CNNBNormModel, self).__init__()
        self.config = config
        self.firstConv = nn.Conv2d(in_channels=1, out_channels=self.config.out_channels, kernel_size=self.config.kernel_size)
        self.firstBNorm = nn.BatchNorm2d(self.config.out_channels)
        h, w = self.compute_out_dim(height=28, width=28, kernel_size=self.config.kernel_size, stride=self.config.conv_stride, padding=0, num_layers=self.config.number_conv_layers)
        for i in range(self.config.number_conv_layers):
            setattr(self, f"conv{i}", nn.Conv2d(in_channels=self.config.out_channels, out_channels=self.config.out_channels * 2 * (i + 1), kernel_size=self.config.kernel_size))
            setattr(self, f"bNorm{i}", nn.BatchNorm2d(self.config.out_channels * 2 * (i + 1)))
            h, w = self.compute_out_dim(height=h, width=w, kernel_size=self.config.kernel_size, stride=self.config.conv_stride,padding=0, num_layers=self.config.number_conv_layers)
        self.fc = nn.Linear(in_features=h * w * self.config.out_channels * 2 * self.config.number_conv_layers, out_features=10)

    def forward(self, result):
        result = functions.relu(self.firstConv(result))
        #print(result.shape, "1")
        for i in range(self.config.number_conv_layers):
            result = getattr(self, f"conv{i}")(result)
            #print(result.shape, "n")
        result = torch.flatten(result, 1)
        #print(result, "2")
        result = self.fc(result)
        return result