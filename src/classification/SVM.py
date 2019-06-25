from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import seaborn as sn

# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer='../data/output/nc16.csv')
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']

print(df.isnull().values.any())

# Optimize hyper-parameters
# hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
# 					'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
# model = svm.SVC()
# model_grid_search = GridSearchCV(model, hyper_params, cv=10, iid=False, n_jobs=-1)
# model_grid_search.fit(X.values, y.values)
# print("Optimal hyper-parameters: ", model_grid_search.best_params_)
# print("Accuracy :", model_grid_search.best_score_)

# Single SVM run
model = svm.SVC(kernel='rbf', gamma=0.001, C=100)
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1)
print(scores)
print(np.mean(scores))
print(np.std(scores))

# Confusion Matrix
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
# model = svm.SVC(kernel='rbf', gamma=0.001, C=100)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print(tn, fp, fn, tp)
# data = {'y_Predicted': y_pred,
#         'y_Actual':    y_test
#         }
# df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, cmap=ListedColormap(['#ED7D31', '#009FDA']), annot=True)
