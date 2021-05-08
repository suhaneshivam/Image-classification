from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K

class LeNet:
    @staticmethod
    def build(channels ,height ,width , numClasses ,activation = "relu" ,weightsPath = None):
        model = Sequential()
        imageShape = (height ,width ,channels)

        if K.image_data_format() == "channels_first":
            imageShape = (channels,height ,width)

        model.add(Conv2D(20 ,kernel_size = 5 ,padding = "same" ,input_shape = imageShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2 ,2) ,strides=(2 ,2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation = activation))
        model.add(Dense(numClasses))
        model.add(Activation(activation = "softmax"))

        if weightsPath is not  None:
            model.load_weights(weightsPath)

        return model
