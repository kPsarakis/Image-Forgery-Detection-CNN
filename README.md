# :mortar_board: Image Forgery Detection with CNNs
In this project, we used [pytorch](https://pytorch.org/) in order to implement a convolutional neural network (CNN) for the purpose of extracting features in the problem of forgery detection. This approach is inspired by the work of Y. Rao et al. [A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911). Then after the feature fusion, proposed in the same paper, we take the extracted features and give them as input to an SVM that does the final binary classification task. The SVM implementation was taken from [scikit-learn](https://scikit-learn.org/stable/). This project was done for the final project of TU Delft's course CS4180 Deep Learning 2019 by Group 10.

## :scroll: System Overview 
The pipeline of the system is:
1. Train the CNN with images close to the distribution that the network will work on.
2. Extract features from unseen images from the same distribution.
3. Use an SVM classifier on the 400 extracted features of the previous step for the final classification.

The pipeline is shown in the following image:
<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/pipeline.png" height="111" width="600">
</p>

## :triangular_ruler: Network Architecture 

The CNN architecture of this project is shown in the image below and is heavily influenced by the [work](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911) of Y. Rao et al. The network structure is 2 convolutions, max pooling, 4 convolutions, 1 max pooling and then 3 convolutions. In the training phase, after the final convolution, a fully connected layer with softmax is applied. In the testing phase, the 400-D output of the final convolutional layer is used in the next *Feature Fusion* step that creates the feature vectors.

<p align="center">
  <img src="https://github.com/kPsarakis/Image-Forgery-Detection-CNN/blob/master/reports/images/network.png" height="331" width="850">
</p>

## :barber: Feature Fusion 


## :flags: Classification with SVM

## :bar_chart: Results

## :office: Project Structure 



## :busts_in_silhouette: Group 10 Team Members 

[Achilleas Vlogiaris](https://github.com/achilleasvlogiaris)

[Arkajit Bhattacharya](https://github.com/arkajitb)

[Kyriakos Psarakis](https://github.com/kPsarakis)

[Panagiotis Soilis](https://github.com/psoilis)

[Rafail Skoulos](https://github.com/RafailSkoulos17)
