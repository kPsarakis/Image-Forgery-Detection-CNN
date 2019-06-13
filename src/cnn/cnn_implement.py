import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
from torch.optim.lr_scheduler import StepLR
from src import filters


class SimpleCNN(nn.Module):

	def __init__(self):
		super(SimpleCNN, self).__init__()

		#Input channels 3 output channels 30
		self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=0)
		# self.conv1.weight = nn.Parameter(filters.get_filters()) - used with SRM filters
		self.conv2 = nn.Conv2d(30, 30, kernel_size=5, stride=2, padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv3 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=0)
		self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
		self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
		self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
		self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
		self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
		self.fc1 = nn.Linear(16*5*5, 1000)
		self.drop1 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(1000, 2)
		
	def forward(self,x):
		x = x.unsqueeze(0)
		# x = F.relu(self.conv1(x)) - used with SRM filters
		x = F.relu(nn.init.xavier_uniform_(self.conv1(x)))
		x = F.relu(nn.init.xavier_uniform_(self.conv2(x)))
		lrn = nn.LocalResponseNorm(2)
		x = lrn(x)
		x = self.pool1(x)
		x = F.relu(nn.init.xavier_uniform_(self.conv3(x)))
		x = F.relu(nn.init.xavier_uniform_(self.conv4(x)))
		x = F.relu(nn.init.xavier_uniform_(self.conv5(x)))
		x = F.relu(nn.init.xavier_uniform_(self.conv6(x)))
		x = lrn(x)
		x = self.pool2(x)
		x = F.relu(nn.init.xavier_uniform_(self.conv7(x)))
		x = F.relu(nn.init.xavier_uniform_(self.conv8(x)))
		x = F.relu(nn.init.xavier_uniform_(self.conv9(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = self.drop1(x)
		x = self.fc2(x)
		return(x)


def outputSize(in_size, kernel_size, stride, padding):
	output = int((in_size-kernel_size+2*(padding))/stride) + 1
	return output


def createLossandOptimizer(net, learning_rate=0.01):
	# TODO: implement early stopping
	loss = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99, weight_decay=5*1e-4)
	return loss, optimizer


def trainNet(net, train_set, n_epochs, learning_rate):
	loss, optimizer = createLossandOptimizer(net, learning_rate)
	scheduler = StepLR(optimizer, step_size=10*len(train_set), gamma=0.9)
	for epoch in range(n_epochs):
		running_loss = 0.0
		print('Epoch: ', epoch + 1)
		print(scheduler.get_lr())
		print(optimizer)
		# loop over the training samples
		for i, data in enumerate(train_set, 0):
			# get the inputs
			inputs, labels = data
			if torch.cuda.is_available():
				inputs = inputs.cuda()
				labels = labels.cuda()
			# zero the parameter gradients
			optimizer.zero_grad()
			# forward + backward + optimize
			outputs = net(inputs)
			labels = [labels]
			labels = torch.from_numpy(np.array(labels))
			loss_size = loss(outputs, labels)
			loss_size.backward()
			# TODO: check why on the 10th epoch it multiplies with an extra 0.9
			scheduler.step()
			# optimizer.step()
			# Print statistics
			running_loss += loss_size.item()
			if i % 2000 == 1999:  # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')


def main():
	dataset = pickle.load(open("dataset.pickle", "rb"))
	train, test = train_test_split(dataset, test_size=0.2, random_state=0, stratify=dataset.targets)
	pickle_out = open("test.pickle", "wb")
	pickle.dump(test, pickle_out)
	pickle_out.close()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if device == "cuda:0":
		CNN = SimpleCNN().cuda()
	else:
		CNN = SimpleCNN()
	trainNet(CNN, train, n_epochs=250, learning_rate=0.01)
	torch.save(CNN.state_dict(), 'Simple_Cnn.pt')


if __name__ == "__main__":
	main()
