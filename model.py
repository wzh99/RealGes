import keras
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential

import gesture
import load
import preproc


class CNN3D(Sequential):
    """
    Hand Gesture Recognition with 3D Convolutional Neural Networks by Molchanov et al.
    """

    def __init__(self):
        # Construct base sequential model
        super().__init__()

        # Main model specification
        # Convolutional layers
        input_shape = [preproc.sequence_length, load.resize_height, load.resize_width, 2]
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


if __name__ == '__main__':
    model = CNN3D()
    opt = keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
    model.compile(opt, loss="categorical_crossentropy", metrics=["accuracy"])
    data_x, data_y = load.load_dataset("data")
    data_y = keras.utils.to_categorical(data_y, len(gesture.category_names))
    model.fit(x=data_x, y=data_y, batch_size=20, epochs=30)
    model.save_weights("cnn3d.h5")
