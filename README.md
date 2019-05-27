# Deep-Learning-Final-Project-Group-10
Final project for TU Delft's course CS4180 Deep Learning 2019

Group 10 members:

[Achilleas Vlogiaris](https://github.com/achilleasvlogiaris)

[Arkajit Bhattacharya](https://github.com/arkajitb)

[Kyriakos Psarakis](https://github.com/kPsarakis)

[Panagiotis Soilis](https://github.com/psoilis)

[Rafail Skoulos](https://github.com/RafailSkoulos17)

## Plan

### Reproduce convolutional network
* Implement the CNN as proposed by Y. Rao et al. 
[A Deep Learning Approach to Detection of Splicing and Copy-Move Forgeries in Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823911).
* Run on the [CASIA v2.0 dataset](https://www.kaggle.com/sophatvathana/casia-dataset).
* Compare its performance with the one of the original paper.

### Run the model on a more "difficult" dataset
* Check the performance of the implemented CNN on the ["difficult" dataset](https://www.nist.gov/itl/iad/mig/media-forensics-challenge) taken from the NIST's Media Forensics Challenge of 2016.
* By difficult we mean that its harder for humans to recognise the fact that an image is tampered. 
However, we understand that this might not be the case for computers and still perform as good as in the "easy" datasets. 
Thus, we decided it would be interesting to find out
### Improve/Fix performance
* Try and make some changes in the architecture in order to improve performance.
* Try out a different convolution layer proposed by B. Bayar et al. [A Deep Learning Approach To Universal Image
Manipulation Detection Using A New Convolutional Layer](https://dl.acm.org/citation.cfm?id=2930786) which is supposed to successfully extract the necessary image 
 features regardless of the forgery method applied.  
* Experiment with different stochastic gradient descent hyperparameters to measure their effect on the 
classification performance. 





    