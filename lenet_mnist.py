from Helper.CNN.Networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

print("[INFO] : Loading MNIST dataset")
((trainX ,trainY) ,(testX ,testY)) = mnist.load_data()
print(trainX.shape ,testX.shape ,trainY.shape ,testY.shape)

if K.image_data_format() == "channels_first":
	trainX = trainX.reshape((trainX.shape[0] ,1 ,28 ,28))
	testX = testX.reshape((testX.shape[0] ,1 ,28 ,28))

else:
	trainX = trainX.reshape((trainX.shape[0] ,28 ,28 ,1))
	testX = testX.reshape((testX.shape[0] ,28 ,28 ,1))

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

print(trainX.shape ,testX.shape ,trainY.shape ,testY.shape)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainY = to_categorical(trainY ,num_classes=10)
testY = to_categorical(testY ,num_classes=10)


print("[INFO]:compiling model")

opt = SGD(lr = 0.01)
model = LeNet.build(1 ,height = 28 ,width = 28 ,numClasses=10 ,weightsPath = args['weights'] if args['load_model'] > 0 else None)
model.compile(optimizer = opt ,loss = "categorical_crossentropy" ,metrics = ["accuracy"])

if args['load_model'] < 0:
	print("[INFO]: Training model")
	H = model.fit(trainX ,trainY ,validation_data = (testX ,testY) ,epochs = 20 ,batch_size = 128 ,verbose = 1)

if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"] ,overwrite=True)

for i in np.random.choice(np.arange(0 ,len(testY)) ,size=(10 ,)):
	probs = model.predict(testX[np.newaxis ,i])
	prediction = probs.argmax(axis = 1)

	image = np.array(testX[i])
	if K.image_data_format == "channels_first":
		image = np.reshape(image ,(image.shape[1] ,image.shape[2]))
		image = (image * 255).astype('uint8')


	else:
		image = np.reshape(image ,(image.shape[0] ,image.shape[1]))
		image = (image * 255).astype('uint8')

	image = cv2.merge([image] * 3)
	image = cv2.resize(image ,(64 ,64) ,interpolation = cv2.INTER_LINEAR)

	cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],np.argmax(testY[i])))
	cv2.imshow("Digit", image)
	cv2.waitKey(0)
