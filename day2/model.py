import torch
from torch import nn

model = nn.Linear(in_features=1, out_features=1, bias=True)


class SingleLayer(nn.Module):

    def __init__(self, in_val):
        super().__init__()
        self.linear = torch.nn.Linear(in_val, 1, bias=True)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


print(model.weight, model.bias)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30 * 5 * 5, out_channels=10, kernel_size=5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)


def main():
    mlp_model = MLP()

    print(list(mlp_model.children()))


if __name__ == "__main__":
    main()
