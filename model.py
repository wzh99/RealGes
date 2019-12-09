import keras
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Activation, Dropout
from keras.layers.pooling import AvgPool3D
from keras.models import Sequential
from typing import Tuple

import os

import gesture

input_length = 32
input_width = 120
input_height = 90

"""
Hand Gesture Recognition with 3D Convolutional Neural Networks by Molchanov et al.
"""


class HRN(Sequential):
    """
    High resolution network.
    """

    def __init__(self):
        # Construct base sequential model
        super().__init__()

        # Main model specification
        # Convolutional layers
        input_shape = [input_length, input_height, input_width, 2]
        self.add(Conv3D(4, (5, 7, 7), input_shape=input_shape))
        self.add(MaxPool3D())
        self.add(Activation("relu"))

        self.add(Conv3D(8, (3, 5, 5)))
        self.add(MaxPool3D())
        self.add(Activation("relu"))

        self.add(Conv3D(32, (3, 5, 5)))
        self.add(MaxPool3D(pool_size=(1, 2, 2)))
        self.add(Activation("relu"))

        self.add(Conv3D(64, (3, 3, 5)))
        self.add(MaxPool3D(pool_size=(1, 2, 2)))
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


class LRN(Sequential):
    """
    Low resolution network.
    """

    def __init__(self):
        super().__init__()

        input_shape = [input_length, input_height, input_width, 2]
        self.add(AvgPool3D(input_shape=input_shape, pool_size=(1, 2, 2)))

        self.add(Conv3D(8, (5, 5, 5)))
        self.add(MaxPool3D())
        self.add(Activation("relu"))

        self.add(Conv3D(32, (3, 5, 5)))
        self.add(MaxPool3D())
        self.add(Activation("relu"))

        self.add(Conv3D(64, (3, 3, 5)))
        self.add(MaxPool3D(pool_size=(2, 2, 4)))
        self.add(Activation("relu"))

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


class C3D(Sequential):
    """
    Learning Spatialtemporal Features with 3D Convolutional Networks by Tran et.al.
    """

    def __init__(self):
        super().__init__()
        
        input_shape = [input_length, input_height, input_width, 2]
        self.add(Conv3D(64, 3, input_shape=input_shape, padding="same", activation="relu"))
        self.add(MaxPool3D())
        
        self.add(Conv3D(128, 3, padding="same", activation="relu"))
        self.add(MaxPool3D())

        self.add(Conv3D(256, 3, padding="same", activation="relu"))
        self.add(Conv3D(256, 3, padding="same", activation="relu"))
        self.add(MaxPool3D())

        self.add(Conv3D(512, 3, padding="same", activation="relu"))
        self.add(Conv3D(512, 3, padding="same", activation="relu"))
        self.add(MaxPool3D())

        self.add(Conv3D(512, 3, padding="same", activation="relu"))
        self.add(Conv3D(512, 3, padding="same", activation="relu"))
        self.add(MaxPool3D())

        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(.5))

        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(.5))

        self.add(Dense(len(gesture.category_names), activation='softmax'))

        opt = keras.optimizers.SGD(learning_rate=0.003, momentum=0.9, nesterov=True)
        self.compile(opt, loss="categorical_crossentropy", metrics=["accuracy"])


network_spec = {
    "hrn": {
        "init": HRN,
        "path": "weights/hrn.h5"
    },
    "lrn": {
        "init": LRN,
        "path": "weights/lrn.h5"
    },
    "c3d": {
        "init": C3D,
        "path": "weights/c3d.h5"
    }
}

def load_model(name: str) -> Tuple[keras.Model]:
    """
    Load one model from file. 
    """
    spec = network_spec[name]
    model = spec["init"]()
    if not os.path.exists(spec["path"]):
        raise RuntimeError("Weight file of model %s is not found." % name)
    model.load_weights(spec["path"])
    return model


def load_two_models(m1_name: str, m2_name: str) -> Tuple[keras.Model, keras.Model]:
    """
    Load two models from file at a time.
    :param m1_name: name of first model to load
    :param m2_name: name of second model to load
    :return: (model_one, model_two)
    """
    m1_model = load_model(m1_name)
    if m1_name == m2_name :
        return m1_model, m1_model
    else:
        m2_model = load_model(m2_name)
        return m1_model, m2_model
