# Image Classification on Animal Dataset
Images Classification task is accomplished using simple feed-forward neural network with an accuracy of 60% and with SmallVggNet which is a variant of original VggNet with reduced number of layers.

#Archutecture
##SmallVggNet
INPUT => CONV => RELU => POOLING => (CONV => RELU) * 2 => POOLING => (CONV => RELU) * 3 => POOLING => FC => RELU => FC => SOFTMAX => OUTPUT

Every block of (CONV => RELU => POOLING) is followed by Batch normalization and occasionally with an additional dropout layer to avoid overfitting.

#Dataset
Dataset contains a total of three categories of animals viz. Dog ,Cat and Panda. Model is trained on
a total of 3000 images ,1000 from each category.

#Training
The Dataset is divided in 75:25 ratio for training and testing purpose which pretty much a standard practice. Relu activation is used at the end of every convolution and fully connected layers. On any standard GPU it takes only 2 seconds per epoch to train the model. The model is trained for 75 epochs in total. For epoch , batch size is kept at 32 images /batch. The SGD(stochastic Gradient Descent) is used to backpropagate the loss gradient. An overall training accuracy of 80 % is achieved at the end of training process.

#Output
The serialized model and labels are kept in output directory for fasted inference on images. Graphs depicting the loss and accuracy of the model on both training and validation is also kept in the same directory of reference.  
