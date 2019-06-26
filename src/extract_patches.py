import os

from patch_extraction.patch_extractor_casia import PatchExtractorCASIA
from patch_extraction.patch_extractor_nc import PatchExtractorNC


# CASIA Dataset
# mode='no_rot' for no rotations
pe = PatchExtractorCASIA(path='../../data/CASIA2_original', patches_per_image=2, stride=32, rotations=4, mode='rot')
pe.extract_patches()

# NC16 Dataset
# mode='no_rot' for no rotations
pe = PatchExtractorNC(path='../../data/NC2016_Test0601/', patches_per_image=2, stride=32, rotations=4, mode='rot')
pe.extract_patches()
