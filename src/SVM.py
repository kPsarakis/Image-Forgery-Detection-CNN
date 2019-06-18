import pandas as pd
import csv
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
print('start------------------------------------------------')
dataset = pd.read_csv(header=None,filepath_or_buffer="dummy_feature_export.csv")
labels = dataset[1].tolist()
dataset = dataset.drop([0,1],axis=1)
list_set = dataset.values.tolist()
svc = svm.SVC(gamma="scale")
print(svc.get_params())
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
					'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()

	clf = GridSearchCV(svc, tuned_parameters, cv=2,
									scoring='%s_macro' % score)
	clf.fit(list_set,labels)

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	        print("%0.3f (+/-%0.03f) for %r"
	              % (mean, std * 2, params))
	print()

	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	y_true, y_pred = labels, clf.predict(list_set)
	print(classification_report(y_true, y_pred))
	print()







