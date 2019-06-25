import torch
import pandas as pd
import torchvision.transforms as transforms
from torchvision import datasets

from cnn.cnn import CNN
from cnn.train_cnn import train_net

torch.manual_seed(0)

DATA_DIR = "my_patches"  # put the directory of the patches in your machine
transform = transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder(root=DATA_DIR, transform=transform)  # Fetch data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if str(device) == "cuda:0":
    print("cuda enabled")
    cnn = CNN().cuda()
else:
    print("no cuda")
    cnn = CNN()

epoch_loss, epoch_accuracy = train_net(cnn, data, n_epochs=250, learning_rate=0.0001, batch_size=128)

pd.DataFrame(epoch_loss).to_csv('SRM_loss.csv')
pd.DataFrame(epoch_accuracy).to_csv('SRM_accuracy.csv')

torch.save(cnn.state_dict(), 'Cnn.pt')
