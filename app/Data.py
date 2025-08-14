from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Data:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_loaders(self):
        train_data = datasets.MNIST(root="./../data", train=True, download=True, transform=self.transform)
        test_data = datasets.MNIST(root="./../data", train=False, download=True, transform=self.transform)

        train_len = int(len(train_data) * 0.8)
        val_len = len(train_data) - train_len

        train_data, val_data = random_split(train_data, [train_len, val_len])

        train_loader = DataLoader(dataset=train_data, batch_size=self.config.batchsize, shuffle=True)
        val_loader = DataLoader(dataset=val_data, batch_size=self.config.batchsize, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=self.config.batchsize, shuffle=True)

        return train_loader, val_loader, test_loader