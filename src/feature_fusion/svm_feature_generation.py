from src.cnn.cnn_backup import SimpleCNN
from src.feature_fusion.feature_fusion import get_Yi, get_Y_hat
import torch
import numpy as np
import pandas as pd
from src.feature_fusion.feature_extraction import get_images_and_labels, get_images_and_labels_nc, get_patches
import torchvision.transforms as transforms
from torch.autograd import Variable
from skimage import io


def create_SVM_features(model):
    transform = transforms.Compose([transforms.ToTensor()])
    df = pd.DataFrame()
    images = get_images_and_labels()
    c = 1
    for image_name in images.keys():  # images
        print("Image: ", c)
        image = images[image_name]['mat']
        label = images[image_name]['label']
        Y = []  # init Y

        patches = get_patches(image, stride=128)

        for patch in patches:  # for every patch
            img_tensor = transform(patch)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor.double())
            Yi = get_Yi(model=model, patch=img_variable)
            Y.append(Yi)  # append Yi to Y

        Y = np.vstack(tuple(Y))

        Y_hat = get_Y_hat(y=Y, operation="mean")  # create Y_hat with mean or max

        df = pd.concat([df, pd.concat([pd.DataFrame([image_name.split("\\")[1], str(label)]), pd.DataFrame(Y_hat)])],
                       axis=1, sort=False)
        c += 1

    # save the feature vector to csv
    final_df = df.T
    final_df.columns = get_df_column_names()
    final_df.to_csv('test.csv', index=False)  # csv type [im_name][label][f1,f2,...,fK]


def create_SVM_features_nc(model):
    transform = transforms.Compose([transforms.ToTensor()])
    df = pd.DataFrame()
    images = get_images_and_labels_nc()
    c = 1
    for image_name, label in images.items():  # images
        print("Image: ", c)
        image = io.imread('../../data/NC2016_Test0601/' + image_name)

        Y = []  # init Y

        patches = get_patches(image, stride=1024)

        for patch in patches:  # for every patch
            img_tensor = transform(patch)
            img_tensor.unsqueeze_(0)
            img_variable = Variable(img_tensor.double())
            Yi = get_Yi(model=model, patch=img_variable)
            Y.append(Yi)  # append Yi to Y

        Y = np.vstack(tuple(Y))

        Y_hat = get_Y_hat(y=Y, operation="mean")  # create Y_hat with mean or max

        df = pd.concat([df, pd.concat([pd.DataFrame([image_name, str(label)]), pd.DataFrame(Y_hat)])],
                       axis=1, sort=False)
        c += 1

    # save the feature vector to csv
    final_df = df.T
    print(get_df_column_names())
    final_df.columns = get_df_column_names()

    final_df.to_csv('test.csv', index=False)  # csv type [im_name][label][f1,f2,...,fK]


def get_df_column_names():
    names = ["image_names", "labels"]
    for i in range(400):
        names.append("f" + str(i + 1))
    return names


def main():
    with torch.no_grad():
        model = SimpleCNN()
        model.load_state_dict(torch.load('../../data/output/CASIA2_Cnn_Full_WithRot_LR0001_b200_nodrop.pt',
                                         map_location=lambda storage, loc: storage))
        model.eval()
        model = model.double()
        create_SVM_features(model)


if __name__ == '__main__':
    main()
