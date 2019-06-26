import pandas as pd
import os


def get_ref_df():
    refs1 = pd.read_csv('../../data/NC2016_Test0601/reference/manipulation/NC2016-manipulation-ref.csv',
                        delimiter='|')
    refs2 = pd.read_csv('../../data/NC2016_Test0601/reference/removal/NC2016-removal-ref.csv', delimiter='|')
    refs3 = pd.read_csv('../../data/NC2016_Test0601/reference/splice/NC2016-splice-ref.csv', delimiter='|')
    all_refs = pd.concat([refs1, refs2, refs3], axis=0)
    return all_refs


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


def check_and_reshape(image, mask):
    if image.shape == mask.shape:
        return image, mask
    elif image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
        mask = np.reshape(mask, (image.shape[0], image.shape[1], mask.shape[2]))
        return image, mask
    else:
        return image, mask
