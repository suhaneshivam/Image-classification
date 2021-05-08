import matplotlib
matplotlib.use('Agg')

from Helper.smallvggnet import SmallVggNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model
import cv2
import imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random
import pickle
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

print("[INFO]: Loading the images")
imagesPaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagesPaths)

images = []
labels = []

for path in imagesPaths:
    image = cv2.imread(path)
    image = cv2.resize(image ,(64 ,64))
    images.append(image)

    label = path.split(os.path.sep)[-2]
    labels.append(label)

images = np.array(images ,dtype="float") /255.0
labels = np.array(labels)

(trainX ,testX ,trainY ,testY) = train_test_split(images ,labels ,test_size = 0.25 ,random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

INIT_LR = 0.01
EPOCHS = 75
BS = 32

aug = ImageDataGenerator(rotation_range=30 ,width_shift_range=0.1 ,
                    height_shift_range=0.1 ,shear_range= 0.2 ,
                    zoom_range=0.2 ,horizontal_flip= True ,fill_mode="nearest")

model = SmallVggNet().build(width = 64 ,height = 64 ,depth = 3 ,classes = len(lb.classes_))
opt = SGD(learning_rate=INIT_LR ,decay=INIT_LR/EPOCHS)
model.compile(optimizer=opt ,loss = "categorical_crossentropy" ,metrics=["accuracy"])

H = model.fit(x = aug.flow(trainX ,trainY ,batch_size= BS) ,
                validation_data=(testX ,testY) ,steps_per_epoch=len(trainX) // BS ,epochs=EPOCHS)
preds = model.predict(x = testX ,batch_size=BS)

print(classification_report(testY.argmax(axis = 1) ,preds.argmax(axis = 1) ,target_names = lb.classes_))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])


model.save(args['model'] ,save_format = 'h5')
with open(args["label_bin"] ,"wb") as f:
    f.write(pickle.dumps(lb))
    f.close()
