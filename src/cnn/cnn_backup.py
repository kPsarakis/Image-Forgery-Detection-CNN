import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import numpy as np
import matplotlib.pyplot as plt

# import filters
from src import filters


class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input channels 3 output channels 30
        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv0.weight)

        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=0)
        # torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1.weight = nn.Parameter(filters.get_filters())  # - used with SRM filters

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv4.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv7.weight)

        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        torch.nn.init.xavier_uniform_(self.conv8.weight)

        self.fc = nn.Linear(16 * 5 * 5, 2)

        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))

        lrn = nn.LocalResponseNorm(3)  # TODO check later

        x = lrn(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = lrn(x)
        x = self.pool2(x)

        x = F.relu(self.conv6(x))

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.view(-1, 16 * 5 * 5)
        # x = self.drop1(x)
        x = F.relu(self.fc(x))  # TODO check later
        x = F.softmax(x, dim=1)

        return x


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output


def createLossandOptimizer(net, learning_rate=0.01):
    # TODO: implement early stopping
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99, weight_decay=5 * 1e-4)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    return loss, optimizer


def trainNet(net, train_set, n_epochs, learning_rate):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10,
                                               shuffle=True, pin_memory=True)
    criterion, optimizer = createLossandOptimizer(net, learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    n_batches = len(train_loader)
    epoch_loss = []
    epoch_accuracy = []
    for epoch in range(n_epochs):
        scheduler.step()
        total_running_loss = 0.0
        print_every = n_batches // 5
        training_start_time = time.time()
        c = 0
        total_predicted = []
        total_labels = []
        # print('Epoch: ', epoch + 1)
        # print(scheduler.get_lr())
        # print(optimizer)
        # loop over the training samples
        for i, (inputs, labels) in enumerate(train_loader):
            # get the inputs
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda().long())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # TODO: check why on the 10th epoch it multiplies with an extra 0.9
            optimizer.step()
            # Print statistics
            # print('Running loss {}'.format(loss_size.item()))
            # print(i)
            # running_loss += loss_size.item()

            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()

            total = labels.size(0)

            total_labels.extend(labels)
            total_predicted.extend(predicted)
            if (i + 1) % (print_every + 1) == 0:  # print every 2000 mini-batches
                total_running_loss += loss.item()
                c += 1
                # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                #       .format(epoch + 1, n_epochs, i + 1, len(train_loader), loss.item(), (correct / total) * 100))

        print('---------- Epoch %d Loss: %.3f  Time: %.3f----------' % (epoch + 1, total_running_loss / c,
                                                                        time.time() - training_start_time))
        epoch_predictions = (np.array(total_predicted) == np.array(total_labels)).sum().item()
        epoch_accuracy.append(epoch_predictions / len(total_predicted))
        epoch_loss.append(total_running_loss / c)
    print('Finished Training')
    return epoch_loss, epoch_accuracy


def plot_epochs(metric, ylab):
    plt.plot(metric)
    plt.ylabel(ylab)
    plt.xlabel("Epoch")
    plt.show()


def main():
    # Image folder with fake and real image subfolders
    torch.manual_seed(0)
    DATADIR = "my_patches"  # put the directory of the file in your machine
    transform = transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    # Fetch data
    dataset = datasets.ImageFolder(root=DATADIR, transform=transform)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda:0":
        print("cuda enabled")
        CNN = SimpleCNN().cuda()
    else:
        print("no cuda")
        CNN = SimpleCNN()
    epoch_loss, epoch_accuracy = trainNet(CNN, dataset, n_epochs=250, learning_rate=0.001)
    plot_epochs(epoch_loss, 'Training Loss')
    plot_epochs(epoch_accuracy, 'Training Accuracy')
    print('Final Loss = {}, Final Accuracy = {}'.format(epoch_loss[-1], epoch_accuracy[-1]))
    torch.save(CNN.state_dict(), 'Simple_Cnn.pt')


if __name__ == "__main__":
    main()
