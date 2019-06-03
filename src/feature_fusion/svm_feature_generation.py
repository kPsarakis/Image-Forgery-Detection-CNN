import time

from src.feature_fusion.feature_extraction import get_images_and_labels, get_patches


def create_SVM_features():
    # for every image (RAFAIL)
    images = get_images_and_labels()
    for image_name in images.keys():
        # get name and label (RAFAIL)
        image = images[image_name]['mat']
        label = images[image_name]['label']

        # generate patches (RAFAIL)
        patches = get_patches(image, stride=128)

        # init Y (KYRIAKOS)

        # for every patch (RAFAIL)
        for patch in patches:
            pass

        # call CNN -> Yi (KYRIAKOS)

        # append Yi to Y (KYRIAKOS)

        # create Y_hat with mean or max (KYRIAKOS)

        # save the feature vector to csv (KYRIAKOS)

        return None


if __name__ == '__main__':
    create_SVM_features()
