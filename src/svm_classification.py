import pandas as pd
from classification.SVM import optimize_hyperparams, classify, print_confusion_matrix

# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer='../data/output/features/CASIA2_WithRot_LR001_b128_nodrop_max_fushion.csv')
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']

print('Has NaN:', df.isnull().values.any())

hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

opt_params = optimize_hyperparams(X, y, params=hyper_params)
classify(X, y, opt_params)
print_confusion_matrix(X, y, opt_params)
