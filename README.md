# Image-Captioning-DL
The image captioning is one of the most important & crucial tasks of the machine learning. This task is accomplished bybusing the prerequisites of Deep learning with the help of python (Anaconda) especially 3.6. This test is originally done in VGG 16 which you can also perform in Inception in the flicker 8k Dataset which you can get from these just by pushing a sign up.
https://forms.illinois.edu/sec/1713398
The required libraries would be:-
Keras 
Theano/Tensorflow
Imgaug
Tqdm
Scipy
Numpy
Rest will be preuploaded by Anaconda.
Support Added-GPU (TENSORFLOW)
Though there is some bug in keras(Backend-Theano) which I will be sorting out in a few days.
Especially the task was created by building CNN & then feeding the data to LSTM network.This is bacsically to provide real time application for the blind persons in need.
This tutorial is divided into 6 parts; they are:
1.	Photo and Caption Dataset
2.	Prepare Photo Data
3.	Prepare Text Data
4.	Develop Deep Learning Model
5.	Train With Progressive Loading (NEW)
6.	Evaluate Model
7.	Generate New Captions


STEP-1

PREPARING TOKENS, IMAGE, & CAPTIONING DATASET.

1. First of all images of different surroundings were used like trees, surroundings, dustbin. The datasets were used. Initially, the data was captured by a phone. Initially, 
the data was taken from a random surroundings. Initially the pixel was of 4130X2130. 

  
After Resizing the image is as follows:-
 
2. After resizing we get the pixels as 512 x 512~540px. The image were initially captured by the Redmi Mobile using f/2 1/33 ISO320, without flash in  general.
3. Dataset-Was created in general by ourselves for the image captioning task which approximately consisted of 100 images taken from different surroundings.
STEP-2
1.	Then after creation of dataset & creation of the taskset & further resizing of set images, we look forward for the creation of labels which were classified as-
[dog,sea,car,dustbin,post,bush,shrub,tree,bench,person,man,woman,cart]
STEP-3
1.	Then, the task of creation of text dataset was done. For each set of images were developed 2 captions along with the reaming of jpg images as a whole.
2.	Then these captions were directly fitted as the labels discussed above.
3.	For each data, i.e. 100 training , testing & validation sets were created as a whole.
•	60- Training.
•	20- Testing.
•	20- Validation , sets were created along with the captions & were given ANNOTATIONS for each set of images. Also, development & tokens.txt files were created as a whole.
Text can be validated for the above image as follows:-

1.jpg#0   There is a car. 1.jpg#1  The car is black in colour. 2.jpg#0  There is one sign board.


STEP-4

DEVELOPING DEEP LEARNING MODEL- 

1.	First, the features were extracted & after the extraction of the features of the photo, features.pkl file was successfully created.
2.	Afterwards, by using keras model basically pretrained  architecture VGG-16 for which the size was resized to 224x224, with the help of RNN+LSTM based model were used was used which were explained with the photo below:-



           

         
The complete model is explained with the help of figure as-
                          
3.	The it is fitted according to the model
The model can be described with the hyperparameters below:
•	Epochs-5
•	Test/Validation Set – 0.2/0.2 % of the total set.
•	Train Set- 0.6% of the total set.
•	Batch_size= 10
•	Verbose=2
•	We got Accuracy of 31.6% on the training Set & 30% on the Testing set.

For which accoding to this after tuning the hyperparameters we got:
•	Epochs-10
•	Test/Validation Set – 0.2/0.2 % of the total set.
•	Train Set- 0.6% of the total set.
•	Batch_size= 15
•	Verbose=2
•	We got Accuracy of 33% on the training Set & 31% on the Testing set.

Then after this , a file ‘model.h5’ was created which is to be used for further tokenization.


STEP-5

CREATING OF TOKENS & EVALUATION OF BLUES SCORE-

1.	After this we fitted the model with the created set of development tokens.
2.	Then a following file named as tokenizer.pkl was being created.
3.	This file was used as a part of further evaluation.
4.	Then the BLUES score were predicted 





