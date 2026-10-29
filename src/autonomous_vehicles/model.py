"""Small original fully convolutional baseline, initialized without pretrained weights."""

from torch import nn


class RoadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, image):
        return self.layers(image)
