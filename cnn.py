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
import torch.optim as optim
import cv2
from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
import time

IMG_SIZE = (128,128)
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
transform = transforms.Compose([transforms.RandomSizedCrop(max(IMG_SIZE)), transforms.ToTensor()])
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
# def create_training_data():
# 	for category in CATEGORIES:
# 		path = os.path.join(DATADIR, category) #path to the directory
# #index the categories
# 		class_num = CATEGORIES.index(category)
# 		for img in os.listdir(path):
# 			try:
# 				img_array = cv2.imread(os.path.join(path,img))
# 				new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
# 				#new_array = Image.fromarray(new_array)
# 				training_data.append([new_array, class_num])
# 			except Exception as e:
# 				pass
# create_training_data()
# print(training_data[0])

trainset = datasets.ImageFolder(root = DATADIR, transform = transform)


train,test = train_test_split(trainset,test_size = 0.2, random_state = 0)

#print(train[0])

class SimpleCNN(nn.Module):

	def __init__(self):
		super(SimpleCNN, self).__init__()

		#Input channels 3 output channels 30
		self.conv1 = nn.Conv2d( 3, 30, kernel_size = 5, stride = 1, padding = 0 )
		self.conv2 = nn.Conv2d( 30, 30, kernel_size = 5, stride = 1, padding = 0 )
		self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
		self.conv3 = nn.Conv2d( 30, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.conv4 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.conv5 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.conv6 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.pool2 = nn.MaxPool2d( kernel_size = 2, stride = 2, padding = 0 )
		self.conv7 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.conv8 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.conv9 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )
		self.conv10 = nn.Conv2d( 16, 16, kernel_size = 3, stride = 1, padding = 0 )

		self.fc1 = nn.Linear(5184, 64)
		self.fc2 = nn.Linear(64, 2)
	def forward(self,x):
		#print("x", x.size())
		x = x.unsqueeze(0)
		#print("x", x.size())
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool1(x)

		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))
		x = self.pool2(x)
		x = F.relu(self.conv7(x))
		x = F.relu(self.conv8(x))
		x = F.relu(self.conv9(x))
		x = F.relu(self.conv10(x))
		x = x.view(-1, 5184)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return(x)

def outputSize(in_size, kernel_size, stride, padding):

	output = int((in_size-kernel_size+2*(padding))/stride) + 1

	return output

def createLossandOptimizer(net, learning_rate = 0.001):
		loss = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		return(loss,optimizer)
def trainNet(net, n_epochs, learning_rate):

	print("===Hyperparameters===")
	print("epochs=", n_epochs)
	print("learning_rate=", learning_rate)
	

	loss,optimizer = createLossandOptimizer(net, learning_rate)
    
	training_start_time = time.time()

	for epoch in range(n_epochs):

		running_loss = 0.0
		start_time = time.time()
		total_train_loss = 0.0
		for i,data in enumerate(train,0):
			inputs,labels = data
			optimizer.zero_grad()

			outputs = net(inputs)
			labels = [labels]
			labels = torch.from_numpy(np.array(labels))

			
			loss_size = loss(outputs,labels)
			loss_size.backward()
			optimizer.step()

			#Print statistics
			running_loss += loss_size.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
				   (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

CNN = SimpleCNN()
trainNet(CNN, n_epochs = 3, learning_rate = 0.001)

torch.save(CNN.state_dict(),'/Users/arkajitbhattacharya/Documents/Pytorch_Project/Simple_Cnn.pt')
#Test Accuracy of the network

def testNet():
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

#print(trainset[0])

# For reversing the operation:
#im_np = np.asarray(im_pil)




#set a standard random seed for reproducible results

