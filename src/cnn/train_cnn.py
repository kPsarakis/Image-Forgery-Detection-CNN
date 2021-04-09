import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
import numpy as np


def create_loss_and_optimizer(net, learning_rate=0.01):
    """
    Creates the loss function and optimizer of the network.
    :param net: The network object
    :param learning_rate: The initial learning rate
    :returns: The loss function and the optimizer
    """
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99, weight_decay=5 * 1e-4)
    return loss, optimizer


def train_net(net, train_set, n_epochs, learning_rate, batch_size):
    """
    Training of the CNN
    :param net: The CNN object
    :param train_set: The training part of the dataset
    :param n_epochs: The number of epochs of the experiment
    :param learning_rate: The initial learning rate
    :param batch_size: The batch size of the SGD
    :returns: The epoch loss (vector) and the epoch accuracy (vector)
    """
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    criterion, optimizer = create_loss_and_optimizer(net, learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    n_batches = len(train_loader)
    epoch_loss = []
    epoch_accuracy = []

    for epoch in range(n_epochs):

        total_running_loss = 0.0
        print_every = n_batches // 5
        training_start_time = time.time()
        c = 0
        total_predicted = []
        total_labels = []

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
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            total_labels.extend(labels)
            total_predicted.extend(predicted)

            if (i + 1) % (print_every + 1) == 0:
                total_running_loss += loss.item()
                c += 1

        epoch_predictions = (np.array(total_predicted) == np.array(total_labels)).sum().item()
        print('---------- Epoch %d Loss: %.3f Accuracy: %.3f Time: %.3f----------' % (
            epoch + 1, total_running_loss / c, epoch_predictions / len(total_predicted),
            time.time() - training_start_time))
        epoch_accuracy.append(epoch_predictions / len(total_predicted))
        epoch_loss.append(total_running_loss / c)
        scheduler.step()

    print('Finished Training')

    return epoch_loss, epoch_accuracy
