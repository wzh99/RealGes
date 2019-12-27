import os
from queue import Queue
from threading import Thread
from typing import List, Tuple

import cv2
import keras
import numpy as np

import gesture
import model
import preproc
from capture import Recorder

# Image viewer
image_size = 500  # size of square images
center = (image_size / 2, image_size / 2)
identity: np.ndarray = cv2.getRotationMatrix2D((0, 0), 0, 1)  # identity matrix
translate_amount = 40  # in pixels
rotate_amount = 30  # in degrees
scale_amount = 1.2


class GestureApp:
    def on_gesture(name: str):
        pass


class ImageViewer(Thread, GestureApp):
    """
    A simple image viewer that demonstrates a possible application of gesture recognition.
    """

    def __init__(self, image_dir: str):
        """
        Constructor
        :param image_dir: path to directory containing at least three images
        """
        super().__init__()
        self.setDaemon(True)  # exit when recorder no longer runs

        # Initialize members
        self.index = 0
        self.queue = Queue(2)  # message queue for asynchronous communication

        self.transform = np.array([identity, identity, identity])

        # Load images from directory
        self.images = []
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (image_size, image_size))
            self.images.append(img)

        # Raise error if there are not enough images
        if len(self.images) < 3:
            raise RuntimeError("%d images found, need at least 3" % (len(self.images)))

        # Define action on each control message
        self.action = {
            "swipe_left": lambda x: self._translate(-translate_amount, 0),
            "swipe_right": lambda x: self._translate(translate_amount, 0),
            "swipe_up": lambda x: self._translate(0, -translate_amount),
            "swipe_down": lambda x: self._translate(0, translate_amount),
            "rotate_ccw": lambda x: self._rotate(rotate_amount),
            "rotate_cw": lambda x: self._rotate(-rotate_amount),
            "expand": lambda x: self._scale(scale_amount),
            "pinch": lambda x: self._scale(1. / scale_amount),
            "one": lambda x: self._set_index(0),
            "two": lambda x: self._set_index(1),
            "three": lambda x: self._set_index(2),
        }

        # Display image
        self._display()

    def _set_index(self, index: int):
        self.index = index

    def _cur_transform(self):
        return self.transform[self.index]

    def _translate(self, x: float, y: float):
        self._cur_transform()[0, 2] += x
        self._cur_transform()[1, 2] += y

    def _rotate(self, deg: float):
        prev_mat = self._to_three_by_three(self._cur_transform())
        rot_mat = self._to_three_by_three(cv2.getRotationMatrix2D(center, deg, 1))
        self._cur_transform()[:, :] = np.matmul(rot_mat, prev_mat)[0:2, 0:3]

    def _scale(self, scale: float):
        prev_mat = self._to_three_by_three(self._cur_transform())
        scale_mat = self._to_three_by_three(cv2.getRotationMatrix2D(center, 0, scale))
        self._cur_transform()[:, :] = np.matmul(scale_mat, prev_mat)[0:2, 0:3]

    @staticmethod
    def _to_three_by_three(affine: np.ndarray):
        mat = np.identity(3)
        mat[0:2, 0:3] = affine
        return mat

    def run(self) -> None:
        while True:
            # Wait for a control message on the queue
            msg = self.queue.get()

            # Decode message
            try:
                self.action[msg](None)
            except KeyError:
                pass

            # Update image
            self._display()

    def _display(self):
        transformed = cv2.warpAffine(self.images[self.index], self._cur_transform(),
                                     (image_size, image_size))
        cv2.imshow("Image Viewer", transformed)

    def on_gesture(self, msg: str) -> None:
        """
        Add control message to queue
        :param msg: message string to be decoded
        """
        self.queue.put_nowait(msg)


def recogize_sample(m1: keras.Model, m2: keras.Model, sample: List[np.ndarray], 
                    app: GestureApp = None):
    """
    Recognize one sample combining result of two models
    :param m1: first model
    :param m2: second model
    :param sample: captured but not normalized gesture sample [depth_chan, grad_chan]
    :param viewer: Image viewer that demonstrate the application of gesture recognition
    """
    # Rescale image sequence
    data = np.ndarray([2, model.input_length, model.input_height, model.input_width])
    for chan_idx in range(2):
        chan = np.ndarray([0, model.input_height, model.input_width])
        for frame_idx in range(len(sample[chan_idx])):
            frame = sample[chan_idx][frame_idx]
            resized = cv2.resize(frame, (model.input_width, model.input_height))
            chan = np.append(chan, [resized], axis=0)
        data[chan_idx] = preproc.temporal_resample(chan, model.input_length)

    # Normalize sequence
    data = np.array([preproc.normalize_sample(data)])

    # Pass to model for prediction result
    result = m1.predict(data) if m1 == m2 else m1.predict(data) * m2.predict(data)
    index = np.argmax(result[0])
    name = gesture.category_names[index]
    print("gesture: %s" % name)
    if app is not None:
        app.on_gesture(name)


if __name__ == '__main__':
    m1, m2 = model.load_two_models("hrn", "hrn")
    viewer = ImageViewer("demo")
    rec = Recorder(callback=lambda seq: recogize_sample(m1, m2, seq, viewer))
    viewer.start()
    rec.record()
