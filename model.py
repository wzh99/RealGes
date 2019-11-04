import keras
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential

import gesture

input_length = 16
input_width = 120
input_height = 90


class CNN3D(Sequential):
    """
    Hand Gesture Recognition with 3D Convolutional Neural Networks by Molchanov et al.
    """

    def __init__(self):
        # Construct base sequential model
        super().__init__()

        # Main model specification
        # Convolutional layers
        input_shape = [input_length, input_height, input_width, 2]
        self.add(Conv3D(4, (5, 7, 7), input_shape=input_shape, data_format="channels_last"))
        self.add(MaxPooling3D())
        self.add(Activation("relu"))

        self.add(Conv3D(8, (3, 5, 5)))
        self.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.add(Activation("relu"))

        self.add(Conv3D(32, (3, 5, 5)))
        self.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.add(Activation("relu"))

        self.add(Conv3D(64, (1, 3, 5)))
        self.add(MaxPooling3D(pool_size=(1, 2, 2)))
        self.add(Activation("relu"))

        # Fully-connected layers
        self.add(Flatten())
        self.add(Dense(512))
        self.add(Activation("relu"))
        self.add(Dropout(0.5))

        self.add(Dense(256))
        self.add(Activation("relu"))
        self.add(Dropout(0.5))

        self.add(Dense(len(gesture.category_names)))
        self.add(Activation("softmax"))

        # Compile models in constructor, since its compiling configuration is fixed
        opt = keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
        self.compile(opt, loss="categorical_crossentropy", metrics=["accuracy"])
