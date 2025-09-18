import torch
import torch.nn as nn
import torch.nn.functional as functions

class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        self.config = config
        self.firstConv = nn.Conv2d(in_channels=1, out_channels=self.config.out_channels, kernel_size=self.config.kernel_size, stride=self.config.conv_stride)
        self.firstBNorm = nn.BatchNorm2d(self.config.out_channels)
        h, w = self.compute_out_dim(height=28, width=28, kernel_size=self.config.kernel_size, stride=self.config.conv_stride, padding=0, num_layers=self.config.number_conv_layers)
        print(w, h, self.config.out_channels)
        in_ch = self.config.out_channels
        out_ch = self.config.out_channels
        for i in range(self.config.number_conv_layers):
            out_ch = self.config.out_channels * 2 * (i + 1)
            setattr(self, f"conv{i}", nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=self.config.kernel_size, stride=self.config.conv_stride))
            setattr(self, f"bNorm{i}", nn.BatchNorm2d(out_ch))
            h, w = self.compute_out_dim(height=h, width=w, kernel_size=self.config.kernel_size, stride=self.config.conv_stride,padding=0, num_layers=self.config.number_conv_layers)
            print(w, h, out_ch)
            in_ch = out_ch
        self.fc = nn.Linear(in_features=w * h * out_ch, out_features=10)
        print(w, h, out_ch)

    def forward(self, result):
        result = functions.relu(self.firstConv(result))
        for i in range(self.config.number_conv_layers):
            result = functions.relu(getattr(self, f"conv{i}")(result))
        result = torch.flatten(result, 1)
        result = self.fc(result)
        return result

    def compute_out_dim(self, height, width, kernel_size, stride, padding, num_layers):
        h, w = height, width
        after_conv_h, after_conv_w = 0, 0
        for _ in range(num_layers):
            after_conv_h = (h + 2 * padding - kernel_size) // stride + 1
            after_conv_w = (w + 2 * padding - kernel_size) // stride + 1

        return after_conv_w, after_conv_h