In case you want to test the pipeline, the following steps need to be taken:

1) Extract CNN training patches: as shown in `extract_patches.py`

2) Train CNN: open the `train_net.py` and change DATA_DIR to point to the patches path extracted from the previous step. Run the script.

3) Extract SVM patches: same as step 1 but with different arguments

4) Compute image features: as shown in `feature_extraction.py`. Here you will need to provide the trained CNN as input.

5) Run SVM cross-validation: change the features path in `svm_classification.py` at line 5 to point to the latest feature extraction. Run the script.
