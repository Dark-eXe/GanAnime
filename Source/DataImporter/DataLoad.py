from .ImageDataset import ImageDataset

import glob
import cv2
from torch.utils.data import DataLoader

class DataLoad(ImageDataset):
    def __init__(self): 
        self.trainDataSet = None
        self.trainDataLoader = None
        self.testDataSet = None
        self.testDataLoader = None
        
    # Preparing Data
    def DataPrep(self, trainingImageFolder, testImageFolder):
        trainingImageFiles = self.LoadingImages(trainingImageFolder)
        testImageFiles = self.LoadingImages(testImageFolder)
        
        self.trainDataSet, self.trainDataLoader = self.CreateTensorData(trainingImageFiles)
        self.testDataSet, self.testDataLoader = self.CreateTensorData(testImageFiles)
    
    # Loading image file names of all jpg/png in directory
    def LoadingImages(self, ImageFolder):
        imageFiles = glob.glob(f"{ImageFolder}/*.jpg") + glob.glob(f"{ImageFolder}/*.png")
        
        return imageFiles
    
    # Creating Tensor Data
    def CreateTensorData(self, imageFiles):
      
        for file in imageFiles:
            image = cv2.imread(file)
            
            if image is None:
                print("Error: Image not found!")
                print(file)
                imageFiles.remove(file)
  
        dataSet = ImageDataset(imageFiles)
        dataLoader = DataLoader(dataSet, batch_size=1, shuffle=True)
            
        return dataSet, dataLoader
        