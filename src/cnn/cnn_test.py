import numpy as np
import torch
from cnn.cnn_implement import SimpleCNN


def testNet():
	CNN = SimpleCNN()
	CNN.load_state_dict(torch.load('Simple_Cnn.pt'))
	CNN.eval()
	correct = 0
	total = 0
	test = torch.load(open("test.pickle", "rb"))
	with torch.no_grad():
		for data in test:
			images, labels = data
			labels = [labels]
			labels = torch.from_numpy(np.array(labels))
			outputs = CNN(images)

			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy : ', (100*correct)/total)


def dummy_test():
	return torch.from_numpy(np.random.randn(5, 5, 16))
