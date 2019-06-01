import torchvision.transforms as transforms
from torchvision import datasets
import pickle


#Crop image based on the image size
IMG_SIZE = (128, 128)

#Image folder with fake and real image subfolders
DATADIR = "../../data/CASIA1";

#Transform variable for tensor
transform = transforms.Compose([transforms.RandomCrop(IMG_SIZE), transforms.ToTensor()])

#Fetch data
dataset = datasets.ImageFolder(root=DATADIR, transform=transform)

#Save the data
pickle_out = open("dataset.pickle", "wb")
pickle.dump(dataset, pickle_out)
pickle_out.close()