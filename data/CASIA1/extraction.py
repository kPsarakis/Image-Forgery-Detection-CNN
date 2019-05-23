import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

save_dir = 'dataset'
Au_pic_list = glob('Au' + os.sep + '*')
Sp_pic_list = glob('Sp' + os.sep + '*')
au_index = [6, 9, 10, 14]
background_index = [14, 21]
foreground_index = [22, 29]

def find_background(Au_pic, Sp_pic_list):
    ### find spliced images with Au_pic as background
    # Au_pic: the path of an authentic image
    # Sp_pic_list: all paths of spliced images
    # result(return): a list of paths with Au_pic as background
    au_name = Au_pic[au_index[0]:au_index[1]] + Au_pic[au_index[2]:au_index[3]]
    backgrounds = []
    for spliced in Sp_pic_list:
        sp_name = spliced[background_index[0]:background_index[1]]
        if au_name == sp_name:
            backgrounds.append(spliced)
    return backgrounds

def splice_save(Au_pic, backgrounds, save_dir):
    # splice together Au_pic and each of backgrounds, and save it/them.
    # Au_pic: the path of an authentic image
    # backgrounds: list returned by `find_background`
    # save_dir: path to save the combined image
    au_image = plt.imread(Au_pic)
    for each in backgrounds:
        sp_image = plt.imread(each)
        if au_image.shape == sp_image.shape:
            result = np.concatenate((au_image, sp_image), 1)
            plt.imsave(save_dir+os.sep+each[14:], result)


    

for Au_pic in Au_pic_list:
    backgrounds = find_background(Au_pic, Sp_pic_list)
    splice_save(Au_pic, backgrounds, save_dir)
    
