import torchvision.transforms.functional as tf
import pandas as pd
import os
import numpy as np
from skimage.util import view_as_windows
from skimage import io
import PIL


def get_ref_df():
    refs1 = pd.read_csv('../../data/NC2016_Test0601/reference/manipulation/NC2016-manipulation-ref.csv',
                        delimiter='|')
    refs2 = pd.read_csv('../../data/NC2016_Test0601/reference/removal/NC2016-removal-ref.csv', delimiter='|')
    refs3 = pd.read_csv('../../data/NC2016_Test0601/reference/splice/NC2016-splice-ref.csv', delimiter='|')
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


def check_and_reshape(image, mask):
    if image.shape == mask.shape:
        return image, mask
    elif image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
        mask = np.reshape(mask, (image.shape[0], image.shape[1], mask.shape[2]))
        return image, mask
    else:
        return image, mask


def extract_all_patches(image, window_shape, stride, num_of_patches, rotations, output_path, im_name, rep_num, mode):
    non_tampered_windows = view_as_windows(image, window_shape, step=stride)
    non_tampered_patches = []
    for m in range(non_tampered_windows.shape[0]):
        for n in range(non_tampered_windows.shape[1]):
            non_tampered_patches += [non_tampered_windows[m][n][0]]
    # select random some patches, rotate and save them
    save_patches(non_tampered_patches, num_of_patches, mode, rotations, output_path, im_name, rep_num)


def save_patches(patches, num_of_patches, mode, rotations, output_path, im_name, rep_num):
    inds = np.random.choice(len(patches), num_of_patches, replace=False)
    if mode == 'rot':
        for i, ind in enumerate(inds):
            for angle in rotations:
                im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(patches[ind][0])), angle=angle,
                                  resample=PIL.Image.BILINEAR)
                im_rt.save(output_path + '/tampered/{0}_{1}_{2}_{3}.png'.format(im_name, i, angle, rep_num))
    else:
        for i, ind in enumerate(inds):
            io.imsave(output_path + '/tampered/{0}_{1}_{2}.png'.format(im_name, i, rep_num), patches[ind][0])


def create_dirs(output_path):
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
