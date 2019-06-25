import pandas as pd
from classification.SVM import optimize_hyperparams, classify, print_confusion_matrix, find_misclassified

# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer='../data/output/features/CASIA2_WithRot_LR001_b128_nodrop.csv')
X = df.loc[:, ~df.columns.isin(['labels', 'image_names'])]
y = df['labels']
img_ids = df['image_names']
print('Has NaN:', df.isnull().values.any())
opt_params = optimize_hyperparams(X, y)
classify(X, y, opt_params)
print_confusion_matrix(X, y, opt_params)
find_misclassified(X, y, opt_params, img_ids)
