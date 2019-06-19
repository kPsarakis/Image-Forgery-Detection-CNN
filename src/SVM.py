from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer="svm_features.csv")
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']

print(df.isnull().values.any())

# Optimize hyper-parameters
# hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
# 					'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
# model = svm.SVC()
# model_grid_search = GridSearchCV(model, hyper_params, cv=10, iid=False)
# model_grid_search.fit(X.values, y.values)
# print("Optimal hyper-parameters: ", model_grid_search.best_params_)
# print("Accuracy :", model_grid_search.best_score_)
# model = svm.SVC(model_grid_search.best_params_)

# Single SVM run
model = svm.SVC(kernel='rbf', gamma=1e-3, C=0.01)
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
print(scores)
print(np.mean(scores))
print(np.std(scores))