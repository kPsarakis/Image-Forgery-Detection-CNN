import torch
from cnn.cnn import CNN
from feature_fusion.feature_vector_generation import create_feature_vectors


with torch.no_grad():
    model = CNN()
    model.load_state_dict(torch.load('../data/output/CASIA2_Cnn_Full_WithRot_LR0001_b200_nodrop.pt',
                                     map_location=lambda storage, loc: storage))
    model.eval()
    model = model.double()
    create_feature_vectors(model)
