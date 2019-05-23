# -*- coding: utf-8 -*-

import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import shutil

'''
提取size一致的图像
'''
extract_dir = ['Au', 'Sp']
for each_dir in extract_dir:    
    pic_list = glob(each_dir+os.sep+'*.jpg') 
    for each in pic_list:
        pic = plt.imread(each)
        if pic.shape==(256,384,3):
            des = 'FixSize\\' + each
            shutil.copy(each, des)


    
    