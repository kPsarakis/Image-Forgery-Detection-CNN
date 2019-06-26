from glob import glob
import PIL
from skimage import io
from skimage.util import view_as_windows
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torchvision.transforms.functional as tf

from patch_extraction.extraction_utils import check_and_reshape, extract_all_patches, create_dirs

warnings.filterwarnings('ignore')
# from src.patch_extraction.mask_extraction import extract_masks


class PatchExtractorCASIA:
    """
    Patch extraction class
    """

    def __init__(self, input_path, output_path, patches_per_image=4, rotations=8, stride=8, mode='no_rot'):
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
        self.mode = mode
        self.input_path = input_path
        self.output_path = output_path

        # define the indices of the image names and read the authentic images
        self.background_index = [13, 21]
        au_index = [3, 6, 7, 12]
        au_pic_list = glob(self.input_path + os.sep + 'Au' + os.sep + '*')
        self.Au_pic_dict = {
            au_pic.split(os.sep)[-1][au_index[0]:au_index[1]] + au_pic.split(os.sep)[-1][au_index[2]:au_index[3]]:
                au_pic for au_pic
            in au_pic_list}

    def extract_authentic_patches(self, sp_pic, num_of_patches, rep_num):
        """
        Extracts and saves the patches from the authentic image
        :param sp_pic: Name of tampered image
        :param num_of_patches: Number of patches to be extracted
        :param rep_num: Number of repetitions being done(just for the patch name)
        """
        sp_name = sp_pic.split('/')[-1][self.background_index[0]:self.background_index[1]]
        if sp_name in self.Au_pic_dict.keys():
            au_name = self.Au_pic_dict[sp_name].split(os.sep)[-1].split('.')[0]
            # define window size
            window_shape = (128, 128, 3)
            au_pic = self.Au_pic_dict[sp_name]
            au_image = plt.imread(au_pic)
            # extract all patches
            extract_all_patches(au_image, window_shape, self.stride, num_of_patches, self.rotations, self.output_path,
                                au_name, rep_num, self.mode)

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
        create_dirs(self.output_path)

        # define window shape
        window_shape = (128, 128, 3)
        tp_dir = self.input_path+'/Tp/'
        rep_num = 0
        # run for all the tampered images
        for f in os.listdir(tp_dir):
            try:
                rep_num += 1
                image = io.imread(tp_dir + f)
                im_name = f.split(os.sep)[-1].split('.')[0]
                # read mask
                mask = io.imread('masks/' + im_name + '_gt.png')
                image, mask = check_and_reshape(image, mask)

                # extract patches from images and masks
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
                if self.mode == 'rot':
                    for i, ind in enumerate(inds):
                        for angle in self.rotations:
                            im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(tampered_patches[ind][0])), angle=angle,
                                              resample=PIL.Image.BILINEAR)
                            im_rt.save(self.output_path+'/tampered/{0}_{1}_{2}_{3}.png'.format(im_name, i, angle,
                                                                                               rep_num))
                else:
                    for i, ind in enumerate(inds):
                        io.imsave(self.output_path+'/tampered/{0}_{1}_{2}.png'.format(im_name, i, rep_num),
                                  tampered_patches[ind][0])
                self.extract_authentic_patches(tp_dir + f, num_of_patches, rep_num)
            except IOError as e:
                rep_num -= 1
                print(str(e))
            except IndexError:
                rep_num -= 1
                print('Mask and image have not the same dimensions')
