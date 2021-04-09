from joblib import load
import torch
from cnn.cnn import CNN
from cv2 import imread
import numpy as np

from src.feature_fusion.feature_vector_generation import get_patch_yi


def get_feature_vector(image_path: str, model):
    feature_vector = np.empty((1, 400))
    feature_vector[0, :] = get_patch_yi(model, imread(image_path))
    return feature_vector


# Load the pretrained CNN with the CASIA2 dataset
with torch.no_grad():
    our_cnn = CNN()
    our_cnn.load_state_dict(torch.load('../data/output/pre_trained_cnn/CASIA2_WithRot_LR001_b128_nodrop.pt',
                                       map_location=lambda storage, loc: storage))
    our_cnn.eval()
    our_cnn = our_cnn.double()

# Load the pretrained svm model
svm_model = load('../data/output/pre_trained_svm/CASIA2_WithRot_LR001_b128_nodrop.pt')

print("Labels are 0 for non-tampered and 1 for tampered")

# Probe the SVM model with a non-tampered image
non_tampered_image_path = '../data/test_images/Au_ani_00002.jpg'
non_tampered_image_feature_vector = get_feature_vector(non_tampered_image_path, our_cnn)
print("Non tampered prediction:", svm_model.predict(non_tampered_image_feature_vector))

# Probe the SVM model with a tampered image
tampered_image_path = '../data/test_images/Tp_D_CNN_M_B_nat00056_nat00099_11105.jpg'
tampered_image_feature_vector = get_feature_vector(tampered_image_path, our_cnn)
print("Tampered prediction:", svm_model.predict(tampered_image_feature_vector))
