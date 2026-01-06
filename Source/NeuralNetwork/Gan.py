from .GenCnn import GenCnn
from .GraderCnn import GraderCnn

import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import matplotlib.pyplot as plt


class Gan():
    def InitializeParameters(self, trainDataLoader, testDataLoader):
        self.INIT_LR = 2e-4
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.W_REG = 0.004
        
        self.lossD = None
        self.lossG = None
        self.randTorch = None
        
        self.trainDataLoader = trainDataLoader
        self.testDataLoader = testDataLoader
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.GenCnn = GenCnn(100, 3).to(self.device)
        self.GraderCnn = GraderCnn(3).to(self.device)
        
        self.lossBce = nn.BCELoss()
        self.optG = Adam(self.GenCnn.parameters(), lr=self.INIT_LR, betas=(0.5, 0.999))
        self.optD = Adam(self.GraderCnn.parameters(), lr=self.INIT_LR, betas=(0.5, 0.999))
        
    
    def TrainNn(self):
        print("Using ", self.device, "for training")
        print("Start Training...")
        
        startTime = time.time()
        
        # Training
        for i in range(self.EPOCHS):
            for self.realImages in self.trainDataLoader:
                self.realImages = self.realImages.to(self.device)
                self.batchSize = self.realImages.size(0)
                
                self.realLabels = torch.ones(self.batchSize, 1, 1, 1, device=self.device)
                self.fakeLabels = torch.zeros(self.batchSize, 1, 1, 1, device=self.device)

                self.__TrainGraderNn()
                self.__TrainGeneratorNn()
            
            print(f"Epoch [{i+1}/{self.EPOCHS}] | D Loss: {self.lossD.item():.4f} | G Loss: {self.lossG.item():.4f}")

            # Save sample of generated images
            if (i + 1) % 10 == 0:
                plt.figure(figsize=(5,5))
                with torch.no_grad():
                    sampleImages = self.GenCnn(torch.randn(16, 100, 1, 1, device=self.device)).cpu()
                sampleImages = (sampleImages + 1) / 2  # Rescale to [0,1]
                grid = torchvision.utils.make_grid(sampleImages, nrow=4)
                plt.imshow(grid.permute(1, 2, 0))
                plt.show()
        
        totalTime = time.time()-startTime
    
        print('Total Training Time: ', round(totalTime, 2), ' seconds\n')

    def GenerateImages(self, num_images=1, show=True) -> torch.Tensor:
        self.GenCnn.eval()

        # latent vectors
        z = torch.randn(num_images, 100, 1, 1, device=self.device)

        with torch.no_grad():
            sample_images = self.GenCnn(z)
            
        # move to CPU and rescale from [-1,1] → [0,1]
        sample_images = (sample_images.cpu() + 1) / 2
        sample_images = sample_images.clamp(0, 1)

        if show:
            grid = torchvision.utils.make_grid(sample_images, nrow=int(num_images**0.5))
            plt.figure(figsize=(5, 5))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.show()
    
        self.GenCnn.train()
        return sample_images

    def SaveModel(self, path="gan.pth") -> None:
        checkpoint = {
            "gen_state_dict": self.GenCnn.state_dict(),
            "disc_state_dict": self.GraderCnn.state_dict(),
            "optG_state_dict": self.optG.state_dict(),
            "optD_state_dict": self.optD.state_dict(),
            "z_dim": 100,
            "image_channels": 3
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def LoadModel(self, path="gan.pth") -> None:
        checkpoint = torch.load(path, map_location=self.device)
    
        # Recreate models (important!)
        self.GenCnn = GenCnn(
            zDim=checkpoint["z_dim"],
            imageChannels=checkpoint["image_channels"]
        ).to(self.device)
    
        self.GraderCnn = GraderCnn(
            imageChannels=checkpoint["image_channels"]
        ).to(self.device)
    
        # Load weights
        self.GenCnn.load_state_dict(checkpoint["gen_state_dict"])
        self.GraderCnn.load_state_dict(checkpoint["disc_state_dict"])
    
        # Optimizers (optional — only if resuming training)
        self.optG.load_state_dict(checkpoint["optG_state_dict"])
        self.optD.load_state_dict(checkpoint["optD_state_dict"])
    
        print("Model loaded successfully")

            
    def __TrainGraderNn(self):
        # Train Discriminator
        self.randTorch = torch.randn(self.batchSize, 100, 1, 1, device=self.device)
        self.fakeImages = self.GenCnn(self.randTorch)
        realLoss = self.lossBce(self.GraderCnn(self.realImages), self.realLabels)
        fakeLoss = self.lossBce(self.GraderCnn(self.fakeImages.detach()), self.fakeLabels)
        self.lossD = realLoss + fakeLoss
        
        self.optD.zero_grad()
        self.lossD.backward()
        self.optD.step()
                
    def __TrainGeneratorNn(self):
        # Train Generator
        self.fakeImages = self.GenCnn(self.randTorch)
        self.lossG = self.lossBce(self.GraderCnn(self.fakeImages), self.realLabels)

        self.optG.zero_grad()
        self.lossG.backward()
        self.optG.step()