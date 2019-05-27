import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_ssim


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


def find_mask(sp_pic, Au_pic_dict):
    '''
    Extracts and saves the mask (ground truth) for the given tampered image.
    :param sp_pic: Tampered image
    :param Au_pic_dict: Dictionary with keys the name of the tampered image and values its path.
    '''
    background_index = [13, 21]  # indices of background image in the tamoered image name
    save_name = sp_pic.split(os.sep)[-1][:-4]  # name of the mask
    sp_name = sp_pic.split(os.sep)[-1][background_index[0]:background_index[1]]
    if sp_name in Au_pic_dict.keys():
        au_pic = Au_pic_dict[sp_name]
        au_image = plt.imread(au_pic)
        sp_image = plt.imread(sp_pic)
        if sp_image.shape == au_image.shape:
            # convert images to grayscale
            gray_au_image = cv2.cvtColor(au_image, cv2.COLOR_BGR2GRAY)
            gray_sp_image = cv2.cvtColor(sp_image, cv2.COLOR_BGR2GRAY)
            # get the difference of the 2 grayscale images
            (score, diff) = compare_ssim(gray_au_image, gray_sp_image, full=True)
            diff = cv2.medianBlur(diff, 1)
            # make background black and tampered area white
            mask = np.ones_like(diff)
            mask[diff < 0.98] = 1
            mask[diff >= 0.98] = 0
            mask = (mask * 255).astype("uint8")
            cv2.imwrite('masks/' + save_name + '_gt.png', mask)


def extract_masks():
    '''
    Extracts and saves all the masks.
    '''
    # create the save directory
    save_dir = 'masks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        delete_prev_images(save_dir)

    # read the authentic and tampered images
    Au_pic_list = glob('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Au' + os.sep + '*')
    Sp_pic_list = glob('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Tp' + os.sep + '*')

    au_index = [3, 6, 7, 12]  # indices of authetic image name

    Au_pic_dict = {
        au_pic.split(os.sep)[-1][au_index[0]:au_index[1]] + au_pic.split(os.sep)[-1][au_index[2]:au_index[3]]:
            au_pic for au_pic
        in Au_pic_list}
    # extract the mask for every tampered image
    for ind, Sp_pic in enumerate(Sp_pic_list):
        find_mask(Sp_pic, Au_pic_dict)


if __name__ == '__main__':
    extract_masks()
