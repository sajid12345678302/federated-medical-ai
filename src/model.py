import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*30*30, 2)
        )

    def forward(self, x):
        return self.net(x)
