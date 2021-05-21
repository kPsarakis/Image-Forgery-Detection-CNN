import torchvision.transforms.functional as tf
import pandas as pd
import os
import numpy as np
from skimage.util import view_as_windows
from skimage import io
import PIL


def get_ref_df():
    """
    Reads the csv files that links the NC2016 images with their masks.
    :returns: All the reference directories
    """
    refs1 = pd.read_csv('../data/NC2016/reference/manipulation/NC2016-manipulation-ref.csv',
                        delimiter='|')
    refs2 = pd.read_csv('../data/NC2016/reference/removal/NC2016-removal-ref.csv', delimiter='|')
    refs3 = pd.read_csv('../data/NC2016/reference/splice/NC2016-splice-ref.csv', delimiter='|')
    all_refs = pd.concat([refs1, refs2, refs3], axis=0)
    return all_refs


def delete_prev_images(dir_name):
    """
    Deletes all the file in a directory.
    :param dir_name: Directory name
    """
    for the_file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def check_and_reshape(image, input_mask):
    """
    Gets an image reshapes it and returns it with its mask.
    :param image: The image
    :param input_mask: The mask of the image
    :returns: the image and its mask
    """
    try:
        mask_x, mask_y = input_mask.shape
        mask = np.empty((mask_x, mask_y, 3))
        mask[:, :, 0] = input_mask
        mask[:, :, 1] = input_mask
        mask[:, :, 2] = input_mask
    except ValueError:
        mask = input_mask
    if image.shape == mask.shape:
        return image, mask
    elif image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
        mask = np.reshape(mask, (image.shape[0], image.shape[1], mask.shape[2]))
        return image, mask
    else:
        return image, mask


def extract_all_patches(image, window_shape, stride, num_of_patches, rotations, output_path, im_name, rep_num, mode):
    """
    Extracts all the patches from an image.
    :param image: The image
    :param window_shape: The shape of the window (for example (128,128,3) in the CASIA2 dataset)
    :param stride: The stride of the patch extraction
    :param num_of_patches: The amount of patches to be extracted per image
    :param rotations: The amount of rotations divided equally in 360 degrees
    :param output_path: The output path where the patches will be saved
    :param im_name: The name of the image
    :param rep_num: The amount of repetitions
    :param mode: If we account rotations 'rot' or nor 'no_rot'
    """
    non_tampered_windows = view_as_windows(image, window_shape, step=stride)
    non_tampered_patches = []
    for m in range(non_tampered_windows.shape[0]):
        for n in range(non_tampered_windows.shape[1]):
            non_tampered_patches += [non_tampered_windows[m][n][0]]
    # select random some patches, rotate and save them
    save_patches(non_tampered_patches, num_of_patches, mode, rotations, output_path, im_name, rep_num,
                 patch_type='authentic')


def save_patches(patches, num_of_patches, mode, rotations, output_path, im_name, rep_num, patch_type):
    """
    Saves all the extracted patches to the output path.
    :param patches: The extracted patches
    :param num_of_patches: The amount of patches to be extracted per image
    :param mode: If we account rotations 'rot' or nor 'no_rot'
    :param rotations: The amount of rotations divided equally in 360 degrees
    :param output_path: The output path where the patches will be saved
    :param im_name: The name of the image
    :param rep_num: The amount of repetitions
    :param patch_type: The mask of the image
    """
    inds = np.random.choice(len(patches), num_of_patches, replace=False)
    if mode == 'rot':
        for i, ind in enumerate(inds):
            image = patches[ind][0] if patch_type == 'tampered' else patches[ind]
            for angle in rotations:
                im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(image)), angle=angle,
                                  resample=PIL.Image.BILINEAR)
                im_rt.save(output_path + '/{0}/{1}_{2}_{3}_{4}.png'.format(patch_type, im_name, i, angle, rep_num))
    else:
        for i, ind in enumerate(inds):
            image = patches[ind][0] if patch_type == 'tampered' else patches[ind]
            io.imsave(output_path + '/{0}/{1}_{2}_{3}.png'.format(patch_type, im_name, i, rep_num), image)


def create_dirs(output_path):
    """
    Creates the directories to the output path.
    :param output_path: The output path
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + '/authentic')
        os.makedirs(output_path + '/tampered')
    else:
        if os.path.exists(output_path + '/authentic'):
            delete_prev_images(output_path + '/authentic')
        else:
            os.makedirs(output_path + '/authentic')
        if os.path.exists(output_path + '/tampered'):
            delete_prev_images(output_path + '/tampered')
        else:
            os.makedirs(output_path + '/tampered')


class NotSupportedDataset(Exception):
    pass


def find_tampered_patches(image, im_name, mask, window_shape, stride, dataset, patches_per_image):
    """
    Gets an image reshapes it and returns it with its mask.
    :param image: The image
    :param im_name: The name of the image
    :param mask: The mask of the image
    :param window_shape: The shape of the window (for example (128,128,3) in the CASIA2 dataset)
    :param stride: The stride of the patch extraction
    :param dataset: The name of the dataset
    :param patches_per_image: The amount of patches to be extracted per image
    :returns: the tampered patches and their amount
    """
    # extract patches from images and masks
    patches = view_as_windows(image, window_shape, step=stride)

    if dataset == 'casia2':
        mask_patches = view_as_windows(mask, window_shape, step=stride)
    elif dataset == 'nc16':
        mask_patches = view_as_windows(mask, (128, 128), step=stride)
    else:
        raise NotSupportedDataset('The datasets supported are casia2 and nc16')

    tampered_patches = []
    # find tampered patches
    for m in range(patches.shape[0]):
        for n in range(patches.shape[1]):
            im = patches[m][n][0]
            ma = mask_patches[m][n][0]
            num_zeros = (ma == 0).sum()
            num_ones = (ma == 255).sum()
            total = num_ones + num_zeros
            if dataset == 'casia2':
                if num_zeros <= 0.99 * total:
                    tampered_patches += [(im, ma)]
            elif dataset == 'nc16':
                if 0.80 * total >= num_ones >= 0.20 * total:
                    tampered_patches += [(im, ma)]

    # if patches are less than the given number then take the minimum possible
    num_of_patches = patches_per_image
    if len(tampered_patches) < num_of_patches:
        print("Number of tampered patches for image {} is only {}".format(im_name, len(tampered_patches)))
        num_of_patches = len(tampered_patches)

    return tampered_patches, num_of_patches
