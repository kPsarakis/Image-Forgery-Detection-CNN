# :mortar_board: Image Forgery Detection with CNNs
In this project, we used [pytorch](https://pytorch.org/) in order to implement a Convolutional Neural Network (CNN) for the purpose of extracting features in the problem of image forgery detection. This approach is inspired by the work of Y. Rao et al. [A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911). Following the feature fusion, proposed in the same paper, we take the extracted features and give them as input to an SVM that performs the final binary classification task. The SVM implementation was taken from [scikit-learn](https://scikit-learn.org/stable/). The datasets used in this project are the [CASIA2](https://www.kaggle.com/sophatvathana/casia-dataset) and the [NC2016](https://www.nist.gov/itl/iad/mig/media-forensics-challenge) datasets. This study was conducted as a final project of TU Delft's course CS4180 Deep Learning 2019 by Group 10.

## :scroll: System Overview 
The pipeline of the system is:
1. Train the CNN with image patches close to the distribution of the images that the network will work on. The training patches contain both tampered and untampered regions from the corresponding images.
2. Extract features from unseen images by breaking them into patches and applying feature fusion after the final convolutional layer of the network.
3. Use an SVM classifier on the 400 extracted features of the previous step for the final classification.

The high-level pipeline is shown in the following image:
<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/pipeline.png" height="111" width="600">
</p>

## :triangular_ruler: Network Architecture 
The CNN architecture of this project is shown in the image below and is influenced by the [work](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911) of Y. Rao et al. The network structure is 2 convolutions, max pooling, 4 convolutions, max pooling and then 3 convolutions. In the training phase, after the final convolution, a fully connected layer with softmax is applied. In the testing phase, the 400-D output of the final convolutional layer is used in the next **Feature Fusion** step that creates the feature vectors.

<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/network.png" height="331" width="850">
</p>

## :barber: Feature Fusion 
In order to create a feature representation of an image during the test phase, *k* patches are extracted and passed through the network. After this procedure, *k* 400-D feature maps are being exported. These feature maps are fused into one feature vector for each image either using max or mean fusion.

## :flags: Classification with SVM
For the final part of the pipeline an SVM classifier is trained and tested using the 400-D representations from the previous step. In particular, we use stratified 10-fold cross-validation to obtain an unbiased error estimate.

## :bar_chart: Results
The accuracy and cross-entropy loss per epoch during the CNN training for the two datasets is shown below:
<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/accuracy_augmented.png" height="300" width="400">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/loss_augmented.png" height="300" width="400">
</p>

The SVM classification accuracy on both datasets after the 10-fold cross-validation is presented in the table below:

| Dataset |    Accuracy    |
| ------- | -------------- |
| CASIA2  | 96.82% ± 1.19% |
| NC2016  | 84.89% ± 6.06% |

For more detailed information feel free to take a look at our project [report](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/Group_10-Image_Forgery_Detection_report.pdf).

## :office: Project Structure [![Codacy Badge](https://api.codacy.com/project/badge/Grade/6913244456df4b9eadf8cae2a34b2e48)](https://www.codacy.com/app/kPsarakis/Image-Forgery-Detection-CNN?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kPsarakis/Image-Forgery-Detection-CNN&amp;utm_campaign=Badge_Grade)
The structure of the project is:

*   [`data`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data) Here lay all the data files related to the project. The CASIA2 and NC16 folders are empty because GitHub does not allow files of such size.
    *   [`output`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output) In this folder we have all the outputs of the pipeline.
        *   [`accuracy`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/accuracy) CSVs containing the accuracy per epoch in all our runs.
        *   [`features`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/features) CSVs containing the final feature representations of every image after the feature fusion part. To minimize the repo size we only maintained two feature files (one per dataset) as an example.
        *   [`loss_function`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/loss_function) CSVs containing the loss per epoch in all our runs.
        *   [`pre_trained_cnn`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/pre_trained_cnn) Pt files that contain the trained CNNs of all our runs.


*   [`reports`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/reports) Final report of the project that contains more details on the implementation.

*   [`src`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src) Source folder of the project. Here we give examples on how to run every part of the pipeline. 
    *   [`classification`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/classification) Folder containing the SVM code.
    *   [`cnn`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/cnn) Folder containing the CNN code.
    *   [`feature_fusion`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/feature_fusion) Folder containing the code used for the feature fusion.
    *   [`patch_extraction`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/patch_extraction) Folder containing the code used for the patch extraction.
    *   [`plots`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/plots) Folder containing the code used for the plots that we generated.

## :busts_in_silhouette: Group 10 Team Members 
[Achilleas Vlogiaris](https://github.com/achilleasvlogiaris)

[Arkajit Bhattacharya](https://github.com/arkajitb)

[Kyriakos Psarakis](https://github.com/kPsarakis)

[Panagiotis Soilis](https://github.com/psoilis)

[Rafail Skoulos](https://github.com/RafailSkoulos17)
