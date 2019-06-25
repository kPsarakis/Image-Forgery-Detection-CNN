import pandas as pd


def get_ref_df():
    refs1 = pd.read_csv('../../data/NC2016_Test0601/reference/manipulation/NC2016-manipulation-ref.csv',
                        delimiter='|')
    refs2 = pd.read_csv('../../data/NC2016_Test0601/reference/removal/NC2016-removal-ref.csv', delimiter='|')
    refs3 = pd.read_csv('../../data/NC2016_Test0601/reference/splice/NC2016-splice-ref.csv', delimiter='|')
    all_refs = pd.concat([refs1, refs2, refs3], axis=0)
    return all_refs
