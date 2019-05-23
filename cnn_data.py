#https://www.youtube.com/watch?v=WvoLTXIjBYU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, Sampler
import pickle


#Crop image based on the image size
IMG_SIZE = (128,128)
#Image folder with fake and real image subfolders
DATADIR = "/Users/arkajitbhattacharya/Documents/Pytorch_Project/casia-dataset/CASIA1";

CATEGORIES = ["Au","Sp"]
#Transform variable for tensor
transform = transforms.Compose([transforms.RandomSizedCrop(max(IMG_SIZE)), transforms.ToTensor()])

#Fetch data
dataset = datasets.ImageFolder(root = DATADIR, transform = transform)

#Save the data
pickle_out = open("dataset.pickle","wb")
pickle.dump(dataset,pickle_out)
pickle_out.close()
#Divide the data to train and test set(20% of data for test)
#train,test = train_test_split(trainset,test_size = 0.2, random_state = 0)

