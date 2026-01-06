from torch.nn import Module
import torch.nn as nn

class GraderCnn(Module):
    def __init__(self, imageChannels):
        super(GraderCnn, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(imageChannels, 32, kernel_size=4, stride=2, padding=1),  # 256x256 → 128x128
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 128x128 → 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 → 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32 → 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16 → 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 8x8 → 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),  # 4x4 → 1x1
            nn.Sigmoid()  # Output probability (real vs. fake)
        )


    def forward(self, x):
        return self.model(x)