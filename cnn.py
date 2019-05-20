#https://www.youtube.com/watch?v=WvoLTXIjBYU
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
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
batch_size = 100
DATADIR = "/Users/arkajitbhattacharya/Documents/Pytorch_Project/casia-dataset/CASIA1";
#transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(128), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#data = datasets.ImageFolder(root=DATADIR,transform = transform)
#loader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)

# image = cv2.imread('/Users/arkajitbhattacharya/Documents/Pytorch_Project/casia-dataset/CASIA1/Au/Au_ani_0001.jpg')
# #plt.imshow(image)
# #plt.show()
# im_pil = Image.fromarray(image)
# print(im_pil)
# #plt.imshow(im_pil)
# #plt.show()
CATEGORIES = ["Au","Sp"]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# for category in CATEGORIES:
# 	path = os.path.join(DATADIR, category) #path to the directory
# 	for img in os.listdir(path):
# 		img_array = cv2.imread(os.path.join(path,img))
# 		img_array_pil = Image.fromarray(img_array)
# 		#plt.imshow(img_array_pil)
# 		#plt.show()
# 		break;
# 	break;
training_data = []
def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category) #path to the directory
#index the categories
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img))
				new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
				new_array = Image.fromarray(new_array)
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
create_training_data()
print(len(training_data))

trainset = datasets.ImageFolder(root = DATADIR, transform = transform)
#print(trainset[1000])

train,test = train_test_split(trainset,test_size = 0.2, random_state = 0)

print(len(train), len(test))


class SimpleCNN(nn.Module):

	def __init__(self):
		super(SimpleCNN, self).__init__()

		#Input channels 3 output channels 30
		self.conv1 = nn.Conv2D(3,30, kernel_size = 5, stride = 1, padding = 0)
		self.conv2 = nn.Conv2D(30,30, kernel_size = 5, stride = 1, padding = 0)
		self.pool = nn.MaxPool2D(kernel_size = 2, stride = 2, padding = 0)
 		self.conv3 = nn.Conv2D(30,16, kernel_size = 3, stride = 1, padding = 0)
 		self.conv4 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		self.conv5 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		self.conv6 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		self.pool = nn.MaxPool2D(kernel_size = 2, stride = 2, padding = 0)
 		self.conv7 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		self.conv8 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		self.conv9 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		self.conv10 = nn.Conv2D(16,16, kernel_size = 3, stride = 1, padding = 0)
 		
 		self.fc1 = nn.Linear(128*128*30, 64)
 		self.fc2 = nn.Linear(64, 2)
    def forward(self,x):
    	x = F.relu(self.conv1(x))

    	x = self.pool(x)
    	x = x.view(-1, 30*128*128)

    	x = F.relu(self.fc1(x))
    	x = self.fc2(x)
    	return(x)

def outputSize(in_size, kernel_size, stride, padding):

	output = int((in_size-kernel_size+2*(padding))/stride) + 1

	return output


#print(trainset[0])

# For reversing the operation:
#im_np = np.asarray(im_pil)




#set a standard random seed for reproducible results

