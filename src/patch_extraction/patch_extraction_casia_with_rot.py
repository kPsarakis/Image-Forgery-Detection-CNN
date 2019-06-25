# -*- coding: utf-8 -*-
from glob import glob

import PIL
from skimage import io
from skimage.util import view_as_windows
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torchvision.transforms.functional as tf

warnings.filterwarnings('ignore')
# from src.patch_extraction.mask_extraction import extract_masks

# define the indices of the image names and read the authentic images
background_index = [13, 21]
au_index = [3, 6, 7, 12]
Au_pic_list = glob('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'CASIA2_original' + os.sep + 'Au' + os.sep + '*')
Au_pic_dict = {
    au_pic.split(os.sep)[-1][au_index[0]:au_index[1]] + au_pic.split(os.sep)[-1][au_index[2]:au_index[3]]:
        au_pic for au_pic
    in Au_pic_list}


class PatchExtractor:
    """
    Patch extraction class
    """

    def __init__(self, patches_per_image=4, rotations=8, stride=8):
        """
        Initialize class
        :param patches_per_image: Number of samples to extract for each image
        :param rotations: Number of rotations to perform
        :param stride: Stride size to be used
        """
        self.patches_per_image = patches_per_image
        self.stride = stride
        rots = [0, 90, 180, 270]
        self.rotations = rots[:rotations]

    @staticmethod
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

    @staticmethod
    def check_and_reshape(image, mask):
        if image.shape == mask.shape:
            return image, mask
        elif image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
            mask = np.reshape(mask, (image.shape[0], image.shape[1], mask.shape[2]))
            return image, mask
        else:
            return image, mask

    def extract_authentic_patches(self, sp_pic, num_of_patches, rep_num):
        """
        Extracts and saves the patches from the authentic image
        :param sp_pic: Name of tampered image
        :param num_of_patches: Number of patches to be extracted
        :param rep_num: Number of repetitions being done(just for the patch name)
        """
        sp_name = sp_pic.split('/')[-1][background_index[0]:background_index[1]]
        if sp_name in Au_pic_dict.keys():
            au_name = Au_pic_dict[sp_name].split(os.sep)[-1].split('.')[0]
            # define window size
            window_shape = (128, 128, 3)
            au_pic = Au_pic_dict[sp_name]
            au_image = plt.imread(au_pic)
            # extract all patches
            non_tampered_windows = view_as_windows(au_image, window_shape, step=self.stride)
            non_tampered_patches = []
            for m in range(non_tampered_windows.shape[0]):
                for n in range(non_tampered_windows.shape[1]):
                    non_tampered_patches += [non_tampered_windows[m][n][0]]
            # select random some patches, rotate and save them
            inds = np.random.choice(len(non_tampered_patches), num_of_patches, replace=False)
            for i, ind in enumerate(inds):
                for angle in self.rotations:
                    im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(non_tampered_patches[ind])), angle=angle,
                                      resample=PIL.Image.BILINEAR)
                    im_rt.save('patches_casia_with_rot/authentic/{0}_{1}_{2}_{3}.png'.format(au_name, i, angle, rep_num))

    def extract_patches(self):
        """
        Main function which extracts all patches
        :return:
        """
        # uncomment to extract masks
        # mask_path = 'masks'
        # if os.path.exists(mask_path) and os.path.isdir(mask_path):
        #     if not os.listdir(mask_path):
        #         print("Extracting masks")
        #         extract_masks()
        #         print("Masks extracted")
        #     else:
        #         print("Masks exist. Patch extraction begins...")
        # else:
        #     os.makedirs(mask_path)
        #     print("Extracting masks")
        #     extract_masks()
        #     print("Masks extracted")

        # create necessary directories
        if not os.path.exists('patches_casia_with_rot'):
            os.makedirs('patches_casia_with_rot')
            os.makedirs('patches_casia_with_rot/authentic')
            os.makedirs('patches_casia_with_rot/tampered')
        else:
            if os.path.exists('patches_casia_with_rot/authentic'):
                self.delete_prev_images('patches_casia_with_rot/authentic')
            else:
                os.makedirs('patches_casia_with_rot/authentic')
            if os.path.exists('patches_casia_with_rot/tampered'):
                self.delete_prev_images('patches_casia_with_rot/tampered')
            else:
                os.makedirs('patches_casia_with_rot/tampered')
        # define window shape
        window_shape = (128, 128, 3)
        tp_dir = '../../data/CASIA2_original/Tp/'
        rep_num = 0
        # run for all the tampered images
        for f in os.listdir(tp_dir):
            try:
                rep_num += 1
                image = io.imread(tp_dir + f)
                im_name = f.split(os.sep)[-1].split('.')[0]
                # read mask
                mask = io.imread('masks/' + im_name + '_gt.png')
                image, mask = self.check_and_reshape(image, mask)

                # extract patches from iamges and masks
                patches = view_as_windows(image, window_shape, step=self.stride)
                mask_patches = view_as_windows(mask, window_shape, step=self.stride)
                tampered_patches = []
                # find tampered patches
                for m in range(patches.shape[0]):
                    for n in range(patches.shape[1]):
                        im = patches[m][n][0]
                        ma = mask_patches[m][n][0]
                        num_zeros = (ma == 0).sum()
                        num_ones = (ma == 255).sum()
                        total = num_ones + num_zeros
                        if num_zeros <= 0.99 * total:
                            tampered_patches += [(im, ma)]
                # if patches are less than the given number then take the minimum possible
                num_of_patches = self.patches_per_image
                if len(tampered_patches) < num_of_patches:
                    print("Number of tampered patches for image {} are only {}".format(f, len(tampered_patches)))
                    num_of_patches = len(tampered_patches)
                # select the best patches, rotate and save them
                inds = np.random.choice(len(tampered_patches), num_of_patches, replace=False)
                for i, ind in enumerate(inds):
                    for angle in self.rotations:
                        im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(tampered_patches[ind][0])), angle=angle,
                                          resample=PIL.Image.BILINEAR)
                        im_rt.save('patches_casia_with_rot/tampered/{0}_{1}_{2}_{3}.png'.format(im_name, i, angle, rep_num))
                self.extract_authentic_patches(tp_dir + f, num_of_patches, rep_num)
            except IOError as e:
                rep_num -= 1
                print(str(e))
            except IndexError:
                rep_num -= 1
                print('Mask and image have not the same dimensions')


if __name__ == '__main__':
    pe = PatchExtractor(patches_per_image=2, stride=32)
    pe.extract_patches()
