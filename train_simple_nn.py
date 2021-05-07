import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d' ,'--dataset' ,required = True ,help = 'path to the image dataset')
ap.add_argument('-m' ,'--model' ,required = True ,help = 'path to output trained model')
ap.add_argument('-b' ,'--label-bin' ,required = True ,help = 'path to output lable binarizer')
ap.add_argument('-p' ,'--plot' ,required = True ,help = 'path to output accuracy/loss plot')
args = vars(ap.parse_args())


print("[INFO] : Loading images")
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)



data = []
labels = []

for path in imagePaths:
    # load the image, resize the image to be 32x32 pixels (ignoring
	# aspect ratio), flatten the image into 32x32x3=3072 pixel image
	# into a list, and store the image in the data list
    image = cv2.imread(path)
    image = cv2.resize(image ,(32 ,32)).flatten()
    data.append(image)

    label = path.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data ,dtype = 'float') / 255.0
labels = np.array(labels)

(trainX ,testX ,trainY ,testY) = train_test_split(data ,labels ,test_size = 0.25 ,random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Sequential()
model.add(Dense(1024 ,input_shape=(3072 ,) ,activation = "sigmoid"))
model.add(Dense(512 ,activation = "sigmoid"))
model.add(Dense(len(lb.classes_) ,activation = "softmax"))

INIT_LR = 0.01
EPOCHS = 80

print("[INFO]: training network")
opt = SGD(lr = INIT_LR)
model.compile(loss = "categorical_crossentropy" ,optimizer=opt ,metrics = ['accuracy'])

H = model.fit(x = trainX ,y = trainY ,validation_data = (testX ,testY) ,epochs = EPOCHS ,batch_size = 32)

print("[INFO]: Evaluating network")
predictions = model.predict(x = testX ,batch_size = 32)
print(classification_report(testY.argmax(axis = 1) ,predictions.argmax(axis = 1) ,target_names=lb.classes_))

N = np.arange(0 ,EPOCHS)
plt.style.use('ggplot')
plt.figure()
plt.plot(N ,H.history['loss'] ,label = "train_loss")
plt.plot(N ,H.history['val_loss'] ,label = "validation_loss")
plt.plot(N ,H.history['accuracy'] ,label = "train_accuracy")
plt.plot(N ,H.history['val_accuracy'] ,label = "validation_accuracy")
plt.title("Training loss and accuracy [simpel NN]")
plt.xlabel("Epoch number")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args['plot'])

print('[INFO]: Serializing network and label binarizer')
model.save(args['model'] ,save_format = 'h5')
f = open(args['label_bin'] ,'wb')
f.write(pickle.dumps(lb))
f.close()
