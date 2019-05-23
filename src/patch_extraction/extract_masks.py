import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import compare_ssim


def delete_prev_images(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def extract_masks(sp_pic, Au_pic_dict):
    save_name = sp_pic.split(os.sep)[-1][:-4]
    sp_name = sp_pic.split(os.sep)[-1][background_index[0]:background_index[1]]
    if sp_name in Au_pic_dict.keys():
        au_pic = Au_pic_dict[sp_name]
        au_image = plt.imread(au_pic)
        sp_image = plt.imread(sp_pic)
        if sp_image.shape == au_image.shape:
            gray_au_image = cv2.cvtColor(au_image, cv2.COLOR_BGR2GRAY)
            gray_sp_image = cv2.cvtColor(sp_image, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(gray_au_image, gray_sp_image, full=True)
            diff = cv2.medianBlur(diff, 1)
            temp = np.ones_like(diff)
            temp[diff < 0.98] = 1
            temp[diff >= 0.98] = 0
            temp = (temp * 255).astype("uint8")
            cv2.imwrite('masks/' + save_name + '_gt.png', temp)


if __name__ == '__main__':
    save_dir = 'masks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        delete_prev_images(save_dir)

    Au_pic_list = glob('..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Au' + os.sep + '*')
    Sp_pic_list = glob('..' + os.sep + 'data' + os.sep + 'CASIA2' + os.sep + 'Tp' + os.sep + '*')
    au_index = [3, 6, 7, 12]
    background_index = [13, 21]
    foreground_index = [22, 29]

    Au_pic_dict = {
        au_pic.split(os.sep)[-1][au_index[0]:au_index[1]] + au_pic.split(os.sep)[-1][au_index[2]:au_index[3]]:
            au_pic for au_pic
        in Au_pic_list}
    for ind, Sp_pic in enumerate(Sp_pic_list):
        extract_masks(Sp_pic, Au_pic_dict)
