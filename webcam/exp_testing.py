import torch.nn as nn
from skimage.io import imread, imshow
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import os
#print(os.getcwd())

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution Layer 1 
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4) 

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2) 
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4)  

        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2) 
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc1 = nn.Linear(64*2*2, 7)
        self.lm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        
        out = out.view(out.size(0), -1) 

        out = self.fc1(out)
        out = self.lm(out)
        
        return out

test_transforms = transforms.Compose([transforms.Grayscale(num_output_channels = 1)
                                                           ,transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
def image_loader(image):
    image = Variable(test_transforms(image))
    image = image.unsqueeze(0) 
    return image 



classes = {0: 'ANGER', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPINESS', 4: 'NEUTRAL', 5: 'SADNESS', 6: 'SURPRISE'}

def TestImage(imgpath):
  image = imread(imgpath)
  #plt.subplot(161),imshow(image)
  image1 = Image.fromarray(image.astype(np.uint8))
  prediction = predict(image1)
  print("prediction is ",classes[prediction])
#TestImage('C:/Users/LAXMI NARSIMHA/Desktop/face recog Testing/happy.jpg')

