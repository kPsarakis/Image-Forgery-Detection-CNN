import numpy as np
import torch
import pickle
from cnn_implement import SimpleCNN

def testNet():
	CNN = SimpleCNN()
	CNN.load_state_dict(torch.load('Simple_Cnn.pt'))
	CNN.eval()
	correct = 0
	total = 0
	test = pickle.load(open("test.pickle", "rb"))
	with torch.no_grad():
		for data in test:
			images, labels = data
			labels = [labels]
			labels = torch.from_numpy(np.array(labels))
			outputs = CNN(images)

			_,predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy : ', (100*correct)/total)

testNet()