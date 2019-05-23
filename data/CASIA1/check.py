# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:21:57 2017

@author: dell
"""
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import shutil

chk_dir = 'Au\\'
pic_list = glob(chk_dir + '*.jpg')

res = []
for each in pic_list:
    pic = plt.imread(each)
    if pic.shape!=(256,384,3):
        print(pic.shape)
        res.append(each)
print(len(res))

'''
# split train and validation set
perm = np.random.permutation(len(pic_list))
train_len = 500

for i in range(train_len):
    des = save_dir+os.sep+'train' + os.sep + pic_list[perm[i]][8:]
    shutil.move(pic_list[perm[i]], des)
'''    
    