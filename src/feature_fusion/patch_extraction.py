import glob
import cv2
from skimage.util import view_as_windows

from ..patch_extraction.extraction_utils import get_ref_df


def get_patches(image_mat, stride):
    """
    Extract patches rom an image
    :param image_mat: The image as a matrix
    :param stride: The stride of the patch extraction process
    :returns: The patches
    """
    window_shape = (128, 128, 3)
    windows = view_as_windows(image_mat, window_shape, step=stride)
    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
            patches += [windows[m][n][0]]
    return patches


def get_images_and_labels(tampered_path, authentic_path):
    """
    Get the images and their corresponding labels
    :param tampered_path: The path containing the tampered images
    :param authentic_path: The path containing the authentic images
    :returns: Dictionary with images and labels
    """
    tampered_dir = tampered_path
    authentic_dir = authentic_path
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
    """
    Get the images and their corresponding labels for the NC 2016 dataset
    :returns: Dictionary with images and labels
    """
    refs = get_ref_df()
    images = {}
    for _, data in refs.iterrows():
        if data['ProbeFileName'] in images:
            continue
        im = data['ProbeFileName']
        images[im] = 1 if data['IsTarget'] == 'Y' else 0
    return images
