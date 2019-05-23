import numpy as np
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, Sampler
import torch.optim as optim
import cv2
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
import time
import pickle
from cnn_implement import SimpleCNN

def testNet():
	CNN = SimpleCNN()
	CNN.load_state_dict(torch.load('/Users/arkajitbhattacharya/Documents/Pytorch_Project/Simple_Cnn.pt'))
	CNN.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in test:
			images,labels = data
			labels = [labels]
			labels = torch.from_numpy(np.array(labels))
			outputs = CNN(images)

			_,predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy : ',(100*correct)/total)

testNet()