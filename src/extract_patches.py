from patch_extraction.patch_extractor_casia import PatchExtractorCASIA
from patch_extraction.patch_extractor_nc import PatchExtractorNC


# CASIA Dataset
# mode='no_rot' for no rotations
pe = PatchExtractorCASIA(input_path='../data/CASIA2', output_path='patches_casia_with_rot',
                         patches_per_image=2, stride=128, rotations=4, mode='rot')
pe.extract_patches()

# NC16 Dataset
# mode='no_rot' for no rotations
# pe = PatchExtractorNC(input_path='../data/NC2016/', output_path='patches_nc_with_rot',
#                       patches_per_image=2, stride=32, rotations=4, mode='rot')
# pe.extract_patches()
