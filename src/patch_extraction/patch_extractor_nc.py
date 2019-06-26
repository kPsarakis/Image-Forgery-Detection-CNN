import torchvision.transforms.functional as tf
import PIL
from skimage import io
from skimage.util import view_as_windows
import os
import numpy as np
import warnings

from patch_extraction.extraction_utils import get_ref_df

# from src.patch_extraction.mask_extraction import extract_masks

warnings.filterwarnings('ignore')


class PatchExtractorNC:
    """
    Patch extraction class
    """

    def __init__(self, path, patches_per_image=4, rotations=8, stride=8, mode='no_rot'):
        """
        Initialize class
        :param path of the dataset
        :param patches_per_image: Number of samples to extract for each image
        :param rotations: Number of rotations to perform
        :param stride: Stride size to be used
        :param mode: patch extraction with or without rotations
        """
        self.patches_per_image = patches_per_image
        self.stride = stride
        rots = [0, 90, 180, 270]
        self.rotations = rots[:rotations]
        self.mode = mode
        self.path = path

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

    def extract_authentic_patches(self, d, num_of_patches, rep_num):
        """
        Extracts and saves the patches from the authentic image
        :param sp_pic: Name of tampered image
        :param num_of_patches: Number of patches to be extracted
        :param rep_num: Number of repetitions being done(just for the patch name)
        """

        # define window size
        window_shape = (128, 128, 3)
        image = io.imread(self.path + d.ProbeFileName)
        # extract all patches
        non_tampered_windows = view_as_windows(image, window_shape, step=self.stride)
        non_tampered_patches = []
        for m in range(non_tampered_windows.shape[0]):
            for n in range(non_tampered_windows.shape[1]):
                non_tampered_patches += [non_tampered_windows[m][n][0]]
        # select random some patches, rotate and save them
        inds = np.random.choice(len(non_tampered_patches), num_of_patches, replace=False)

        if self.mode == 'rot':
            for i, ind in enumerate(inds):
                for angle in self.rotations:
                    im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(non_tampered_patches[ind])), angle=angle,
                                      resample=PIL.Image.BILINEAR)
                    im_rt.save('patches_nc_rot/authentic/{0}_{1}_{2}_{3}.png'
                               .format(d.ProbeFileName.split('.')[-2].split('/')[-1], i, angle, rep_num))
        else:
            for i, ind in enumerate(inds):
                io.imsave('patches_nc_no_rot/authentic/{0}_{1}.png'
                          .format(d.ProbeFileName.split('.')[-2].split('/')[-1], i), non_tampered_patches[ind])

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

        all_refs = get_ref_df()

        # create necessary directories
        if not os.path.exists('patches_nc_rot'): # TODO Check the dirs
            os.makedirs('patches_nc_rot')
            os.makedirs('patches_nc_rot/authentic')
            os.makedirs('patches_nc_rot/tampered')
        else:
            if os.path.exists('patches_nc_rot/authentic'):
                self.delete_prev_images('patches_nc_rot/authentic')
            else:
                os.makedirs('patches_nc_rot/authentic')
            if os.path.exists('patches_nc_rot/tampered'):
                self.delete_prev_images('patches_nc_rot/tampered')
            else:
                os.makedirs('patches_nc_rot/tampered')
        # define window shape
        window_shape = (128, 128, 3)
        mask_window_shape = (128, 128)
        rep_num = 0
        # run for all the tampered images
        images_checked = {}
        for i, d in all_refs.iterrows():
            if d.ProbeFileID in images_checked:
                continue
            else:
                images_checked[d.ProbeFileID] = 1
            if d.IsTarget == 'Y':
                try:
                    image = io.imread(self.path + d.ProbeFileName)
                    mask = io.imread(self.path + d.ProbeMaskFileName)
                    rep_num += 1
                    image, mask = self.check_and_reshape(image, mask)

                    # extract patches from images and masks
                    patches = view_as_windows(image, window_shape, step=self.stride)
                    mask_patches = view_as_windows(mask, mask_window_shape, step=self.stride)
                    tampered_patches = []
                    # find tampered patches
                    for m in range(patches.shape[0]):
                        for n in range(patches.shape[1]):
                            im = patches[m][n][0]
                            ma = mask_patches[m][n][0]
                            num_zeros = (ma == 0).sum()
                            num_ones = (ma == 255).sum()
                            total = num_ones + num_zeros
                            if 0.80 * total >= num_ones >= 0.20 * total:
                                tampered_patches += [(im, ma)]
                    # if patches are less than the given number then take the minimum possible
                    num_of_patches = self.patches_per_image
                    if len(tampered_patches) < num_of_patches:
                        print("Number of tampered patches for image are only {}".format(len(tampered_patches)))
                        num_of_patches = len(tampered_patches)
                    # select the best patches, rotate and save them
                    inds = np.random.choice(len(tampered_patches), num_of_patches, replace=False)
                    if self.mode == 'rot':
                        for i, ind in enumerate(inds):
                            for angle in self.rotations:
                                im_rt = tf.rotate(PIL.Image.fromarray(np.uint8(tampered_patches[ind][0])), angle=angle,
                                                  resample=PIL.Image.BILINEAR)
                                im_rt.save('patches_nc_rot/tampered/{0}_{1}_{2}_{3}.png'.format(
                                    d.ProbeFileName.split('.')[-2].split('/')[-1], i, angle, rep_num))
                    else:
                        for i, ind in enumerate(inds):
                            io.imsave('patches_nc_no_rot/tampered/{0}_{1}.png'.format(
                                d.ProbeFileName.split('.')[-2].split('/')[-1], i), tampered_patches[ind][0])
                except IOError as e:
                    rep_num -= 1
                    print(str(e))
                except IndexError:
                    rep_num -= 1
                    print('Mask and image have not the same dimensions')
            else:
                self.extract_authentic_patches(d, self.patches_per_image, rep_num)
