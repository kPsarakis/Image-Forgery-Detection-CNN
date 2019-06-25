import glob
import pandas as pd
import cv2
from skimage.util import view_as_windows


def get_images_and_labels():
    tampered_dir = '../../data/CASIA2_original/Tp/*'
    authentic_dir = '../../data/CASIA2_original/Au/*'
    images = {}
    for im in glob.glob(authentic_dir):
        images[im] = {}
        images[im]['mat'] = cv2.imread(im)
        images[im]['label'] = 0
    for im in glob.glob(tampered_dir):
        images[im] = {}
        images[im]['mat'] = cv2.imread(im)
        images[im]['label'] = 1
    return images


def get_images_and_labels_nc():
    refs = get_ref_df()
    images = {}
    for index, data in refs.iterrows():
        if data['ProbeFileName'] in images:
            continue
        im = data['ProbeFileName']
        images[im] = 1 if data['IsTarget'] == 'Y' else 0
    return images


def get_patches(image_mat, stride):
    window_shape = (128, 128, 3)
    windows = view_as_windows(image_mat, window_shape, step=stride)
    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
            patches += [windows[m][n][0]]
    return patches


def get_ref_df():
    refs1 = pd.read_csv('../../data/NC2016_Test0601/reference/manipulation/NC2016-manipulation-ref.csv',
                        delimiter='|')
    refs2 = pd.read_csv('../../data/NC2016_Test0601/reference/removal/NC2016-removal-ref.csv', delimiter='|')
    refs3 = pd.read_csv('../../data/NC2016_Test0601/reference/splice/NC2016-splice-ref.csv', delimiter='|')
    all_refs = pd.concat([refs1, refs2, refs3], axis=0)
    return all_refs
