# -*- coding: utf-8 -*-

import numpy as np
from skimage import io
from sklearn.feature_extraction.image import extract_patches_2d
import os


def delete_prev_images(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def extract_patches():
    if not os.path.exists('patches'):
        os.makedirs('patches')
        os.makedirs('patches/authentic')
        os.makedirs('patches/tampered')
    else:
        if os.path.exists('patches/authentic'):
            delete_prev_images('patches/authentic')
        else:
            os.makedirs('patches/authentic')
        if os.path.exists('patches/tampered'):
            delete_prev_images('patches/tampered')
        else:
            os.makedirs('patches/tampered')

    window_shape = (128, 128)
    tp_dir = '../data/CASIA2/Tp/'
    j = 0
    for f in os.listdir(tp_dir):
        try:
            image = io.imread(tp_dir + f)
            im_name = f.split(os.sep)[-1].split('.')[0]
            mask = io.imread('masks/' + im_name + '_gt.png')
            patches = extract_patches_2d(image, window_shape)
            mask_patches = extract_patches_2d(mask, window_shape)
            # patches = view_as_windows(A, window_shape)
            non_tampered_patches = []
            tampered_patches = []
            for im, ma in zip(patches, mask_patches):
                num_zeros = (ma == 0).sum()
                num_ones = (ma == 255).sum()
                total = num_ones + num_zeros
                if 0.8 * total >= num_zeros >= 0.2 * total and 0.8 * total >= num_ones >= 0.2 * total and num_ones:
                    tampered_patches += [im]
                elif num_zeros >= 0.95 * total:
                    non_tampered_patches += [im]
            num_of_patches = 100
            if len(non_tampered_patches) < num_of_patches or len(tampered_patches) < num_of_patches:
                num_of_patches = min(len(non_tampered_patches), len(tampered_patches))

            inds = np.random.choice(len(non_tampered_patches), num_of_patches, replace=False)
            for i, ind in enumerate(inds):
                io.imsave('patches/authentic/patch_{0}_{1}.png'.format(j, i), non_tampered_patches[ind])

            inds = np.random.choice(len(tampered_patches), num_of_patches, replace=False)
            for i, ind in enumerate(inds):
                io.imsave('patches/tampered/patch_{0}_{1}.png'.format(j, i), tampered_patches[ind])
            j += 1
        except OSError as e:
            print(e.message)
            pass


if __name__ == '__main__':
    extract_patches()
