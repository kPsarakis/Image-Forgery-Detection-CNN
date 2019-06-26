# :mortar_board: Image Forgery Detection with CNNs
In this project, we used [pytorch](https://pytorch.org/) in order to implement a convolutional neural network (CNN) for the purpose of extracting features in the problem of forgery detection. This approach is inspired by the work of Y. Rao et al. [A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911). Then after the feature fusion, proposed in the same paper, we take the extracted features and give them as input to an SVM that does the final binary classification task. The SVM implementation was taken from [scikit-learn](https://scikit-learn.org/stable/). The datasets used in this project are the [CASIA2](https://www.kaggle.com/sophatvathana/casia-dataset) and the [NC2016](https://www.nist.gov/itl/iad/mig/media-forensics-challenge) datasets. This project was done for the final project of TU Delft's course CS4180 Deep Learning 2019 by Group 10.

## :scroll: System Overview 
The pipeline of the system is:
1. Train the CNN with image paches close to the distribution of the images that the network will work on. The training patches contain both tampered and untampered regions from the tampered images and randomly selected ones from the untampered images.
2. Extract features from unseen images by breaking them into patches and applying feature fusion after the final convolutional layer of the network.
3. Use an SVM classifier on the 400 extracted features of the previous step for the final classification.

The pipeline is shown in the following image:
<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/pipeline.png" height="111" width="600">
</p>

## :triangular_ruler: Network Architecture 
The CNN architecture of this project is shown in the image below and is heavily influenced by the [work](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911) of Y. Rao et al. The network structure is 2 convolutions, max pooling, 4 convolutions, 1 max pooling and then 3 convolutions. In the training phase, after the final convolution, a fully connected layer with softmax is applied. In the testing phase, the 400-D output of the final convolutional layer is used in the next **Feature Fusion** step that creates the feature vectors.

<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/network.png" height="331" width="850">
</p>

## :barber: Feature Fusion 
In order to create a feature representation of an image during test phase, *k* patches are extracted and passed though the network. After this procedure, *k* 400-D feature maps are being exported. Finally, these feature maps are fused into one either by a max or mean opperation.

## :flags: Classification with SVM
For the final part of the pipeline an SVM classifier is constructed and the accuracy estimation is calculated by using 10-fold cross-validation.

## :bar_chart: Results
The accuracy results on both datasets after 10-fold cross-validation during the SVM classification are showcased in the table below:

| Dataset | Accuracy |
| --------| -------- |
| CASIA2  | 96.82% ± 1.19% |
| NC2016  | 84.89% ± 6.06% |

The accuracy per epoch during the training phase of the CNN is shown below for the 2 datasets:
<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/accuracy_augmented.png" height="300" width="400">
</p>

The loss, mesured by the cross-entropy loss, per epoch during the training phase of the CNN is shown below for the 2 datasets:
<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/loss_augmented.png" height="300" width="400">
</p>

For more information feel free to take a look at our [report](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/Group_10-Image_Forgery_Detection_report.pdf).

## :office: Project Structure [![Codacy Badge](https://api.codacy.com/project/badge/Grade/6913244456df4b9eadf8cae2a34b2e48)](https://www.codacy.com/app/kPsarakis/Image-Forgery-Detection-CNN?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kPsarakis/Image-Forgery-Detection-CNN&amp;utm_campaign=Badge_Grade)
The structure of the project is:

* [`data`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data) Here lie all the data files related to the project. The CASIA2 and NC16 folders are empty because GitHub does not allow files of such size.
  * [`output`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output) In this folder we have all the outputs of the pipeline.
    * [`accuracy`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/accuracy)
    * [`features`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/features)
    * [`loss_function`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/loss_function)
    * [`pre_trained_cnn`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/data/output/pre_trained_cnn)
* [`reports`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/reports)
* [`src`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src)
  * [`classification`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/classification)
  * [`cnn`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/cnn)
  * [`feature_fusion`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/feature_fusion)
  * [`patch_extraction`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/patch_extraction)
  * [`plots`](https://github.com/kPsarakis/Image-Forgery-Detection-CNN/tree/master/src/plots)

## :busts_in_silhouette: Group 10 Team Members 
[Achilleas Vlogiaris](https://github.com/achilleasvlogiaris)

[Arkajit Bhattacharya](https://github.com/arkajitb)

[Kyriakos Psarakis](https://github.com/kPsarakis)

[Panagiotis Soilis](https://github.com/psoilis)

[Rafail Skoulos](https://github.com/RafailSkoulos17)
