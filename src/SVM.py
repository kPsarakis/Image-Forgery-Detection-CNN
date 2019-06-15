import pandas as pd
import csv
from sklearn import svm
import numpy as np
import pandas as pd

dataset = pd.read_csv(header=None,filepath_or_buffer="dummy_feature_export.csv")
labels = dataset[1].tolist()
dataset = dataset.drop([0,1],axis=1)
list_set = dataset.values.tolist()
clf = svm.SVC()
clf.fit(list_set,labels)
