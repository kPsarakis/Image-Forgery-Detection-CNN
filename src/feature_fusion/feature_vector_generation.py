import os

from src.feature_fusion.feature_fusion import get_yi, get_y_hat

import numpy as np
import pandas as pd
from src.feature_fusion.patch_extraction import get_images_and_labels, get_images_and_labels_nc, get_patches
import torchvision.transforms as transforms
from torch.autograd import Variable
from skimage import io


def create_feature_vectors(model, tampered_path, authentic_path, output_name):
    """
    Writes the feature vectors of the CASIA2 dataset.
    :param model: The pre-trained CNN object
    :param tampered_path: The path of the tampered images of the CASIA2 dataset
    :param authentic_path: The path of the authentic images of the CASIA2 dataset
    :param output_name: The name of the output CSV that contains the feature vectors
    """
    df = pd.DataFrame()
    images = get_images_and_labels(tampered_path, authentic_path)
    c = 1
    for image_name in images.keys():  # images
        print("Image: ", c)

        image = images[image_name]['mat']
        label = images[image_name]['label']

        df = pd.concat([df, pd.concat([pd.DataFrame([image_name.split(os.sep)[-1], str(label)]),
                                       pd.DataFrame(get_patch_yi(model, image))])], axis=1, sort=False)
        c += 1

    # save the feature vector to csv
    final_df = df.T
    final_df.columns = get_df_column_names()
    final_df.to_csv(output_name, index=False)  # csv type [im_name][label][f1,f2,...,fK]


def create_feature_vectors_nc(model, input_path, output_name):
    """
    Writes the feature vectors of the NC 2016 dataset.
    :param model: The pre-trained CNN object
    :param input_path: The path of the dataset
    :param output_name: The name of the output CSV that contains the feature vectors
    """
    df = pd.DataFrame()
    images = get_images_and_labels_nc()
    c = 1
    for image_name, label in images.items():  # images

        print("Image: ", c)

        image = io.imread(input_path + image_name)

        df = pd.concat([df, pd.concat([pd.DataFrame([image_name, str(label)]),
                                       pd.DataFrame(get_patch_yi(model, image))])], axis=1, sort=False)
        c += 1

    # save the feature vector to csv
    final_df = df.T
    final_df.columns = get_df_column_names()
    final_df.to_csv(output_name, index=False)  # csv type [im_name][label][f1,f2,...,fK]


def get_patch_yi(model, image):
    """
    Calculates the feature representation of an image.
    :param model: The pre-trained CNN object
    :param image: The image
    :returns: The image's feature representation
    """
    transform = transforms.Compose([transforms.ToTensor()])

    y = []  # init Y

    patches = get_patches(image, stride=1024)

    for patch in patches:  # for every patch
        img_tensor = transform(patch)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor.double())
        yi = get_yi(model=model, patch=img_variable)
        y.append(yi)  # append Yi to Y

    y = np.vstack(tuple(y))

    y_hat = get_y_hat(y=y, operation="mean")  # create Y_hat with mean or max

    return y_hat


def get_df_column_names():
    """
    Rename the feature csv column names as [im_names][labels][f1,f2,...,fK].
    :returns: The column names
    """
    names = ["image_names", "labels"]
    for i in range(400):
        names.append("f" + str(i + 1))
    return names
