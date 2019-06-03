# -*- coding: utf-8 -*-
import functools
from glob import glob
from skimage import io
from skimage.util import view_as_windows
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')
# from src.patch_extraction.mask_extraction import extract_masks

# define the indices of the image names and read the authentic images
background_index = [13, 21]
au_index = [3, 6, 7, 12]
Au_pic_list = glob('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Au' + os.sep + '*')
Au_pic_dict = {
    au_pic.split(os.sep)[-1][au_index[0]:au_index[1]] + au_pic.split(os.sep)[-1][au_index[2]:au_index[3]]:
        au_pic for au_pic
    in Au_pic_list}


class PatchExtractor:
    '''
    Patch extraction class
    '''

    def __init__(self, patches_per_image=4, rotations=8, stride=8):
        '''
        Initialize class
        :param patches_per_image: Number of samples to extract for each image
        :param rotations: Number of rotations to perform
        :param stride: Stride size to be used
        '''
        self.patches_per_image = patches_per_image
        self.stride = stride
        rots = [0, 90, 180, 270, 45, 135, 225, 315]
        self.rotations = rots[:rotations]

    @staticmethod
    def delete_prev_images(dir):
        '''
        Deletes all the file in a directory.
        :param dir: Directory name
        '''
        for the_file in os.listdir(dir):
            file_path = os.path.join(dir, the_file)
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

    @staticmethod
    def rotate_image(mat, angle):
        '''
        Rotates an image (angle in degrees) and expands image to avoid cropping
        :param mat: Matrix representation of image
        :param angle: Angle of rotation
        :return: Matrix representation of rotated image
        '''
        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (
            width / 2,
            height / 2)
        # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def extract_authentic_patches(self, sp_pic, num_of_patches, rep_num):
        '''
        Extracts and saves the patches from the authentic image
        :param sp_pic: Name of tampered image
        :param num_of_patches: Number of patches to be extracted
        :param rep_num: Number of repetitions being done(just for the patch name)
        '''
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
                    img_rt = self.rotate_image(non_tampered_patches[ind], angle)
                    io.imsave('patches/authentic/{0}_{1}_{2}_{3}.png'.format(au_name, i, angle, rep_num), img_rt)

    def extract_patches(self):
        '''
        Main function which extracts all patches
        :return:
        '''
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
        if not os.path.exists('patches'):
            os.makedirs('patches')
            os.makedirs('patches/authentic')
            os.makedirs('patches/tampered')
        else:
            if os.path.exists('patches/authentic'):
                self.delete_prev_images('patches/authentic')
            else:
                os.makedirs('patches/authentic')
            if os.path.exists('patches/tampered'):
                self.delete_prev_images('patches/tampered')
            else:
                os.makedirs('patches/tampered')
        # define window shape
        window_shape = (128, 128, 3)
        tp_dir = '../../data/CASIA2/Tp/'
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
                        img_rt = self.rotate_image(tampered_patches[ind][0], angle)
                        io.imsave('patches/tampered/{0}_{1}_{2}_{3}.png'.format(im_name, i, angle, rep_num), img_rt)
                self.extract_authentic_patches(tp_dir + f, num_of_patches, rep_num)
            except IOError as e:
                rep_num -= 1
                print(str(e))
                pass
            except IndexError as ie:
                rep_num -= 1
                print('Mask and image have not the same dimensions')
                pass


if __name__ == '__main__':
    pe = PatchExtractor(patches_per_image=2, stride=32)
    pe.extract_patches()
