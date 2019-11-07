import os
from typing import Tuple

import cv2
import numpy as np
import h5py

import gesture
import model
import preproc


def from_directory(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the whole dataset from directory and assemble as channel first training data.
    :param path: where to load dataset
    :return: (data_x, data_y)
        data_x.shape: [num_samples, num_channels, seq_len, image_height, image_width]
        data_x.dtype: numpy.uint8
        data_y.shape: [num_samples]
        data_y.dtype: numpy.int
    """
    data_x = np.ndarray([0, 2, model.input_length, model.input_height, model.input_width],
                        dtype=np.uint8)
    data_y = np.ndarray([0], dtype=np.int)

    for gesture_id in range(len(gesture.category_names)):
        gesture_dir = os.path.join(path, gesture.category_names[gesture_id])
        if not os.path.exists(gesture_dir):
            continue
        for seq_name in os.listdir(gesture_dir):
            seq_dir = os.path.join(gesture_dir, seq_name)
            seq = _load_one_sequence(seq_dir)
            data_x = np.append(data_x, [seq], axis=0)
            data_y = np.append(data_y, gesture_id)

    print("data_x:", data_x.shape, "data_y:", data_y.shape)
    return data_x, data_y


def _load_one_sequence(path: str) -> np.ndarray:
    """
    Load one gesture sequence from file and normalize it.
    :param path: where to load a single sequence
    :return: normalized gesture sample
        shape: [num_channels, sequence_len, image_height, image_width]
        dtype: numpy.float32
    """
    # Load depth images and find sequence length
    depth_seq = []
    seq_len = 0
    while True:
        depth_file = os.path.join(path, "d%02d.jpg" % seq_len)
        if not os.path.exists(depth_file):
            break
        depth_image = cv2.imread(depth_file, flags=cv2.IMREAD_UNCHANGED)
        depth_image = cv2.resize(depth_image, (model.input_width, model.input_height))
        depth_seq.append(depth_image)
        seq_len += 1
    depth_seq = preproc.temporal_resample(np.array(depth_seq), model.input_length)

    # Load gradient images
    grad_seq = []
    for i in range(seq_len):
        grad_file = os.path.join(path, "g%02d.jpg" % i)
        grad_img = cv2.imread(grad_file, flags=cv2.IMREAD_UNCHANGED)
        grad_img = cv2.resize(grad_img, (model.input_width, model.input_height))
        grad_seq.append(grad_img)
    grad_seq = preproc.temporal_resample(np.array(grad_seq), model.input_length)

    return np.array([depth_seq, grad_seq])


def _store_as_hdf5(path: str, data_x: np.ndarray, data_y: np.ndarray) -> None:
    """
    Store dataset as an HDF5 file
    :param data_x: gesture sample array
        shape: [num_samples, num_channels, seq_len, image_height, image_width]
        dtype: numpy.uint8
    :param data_y: gesture index label (not categorical)
        shape: [num_samples]
        dtype: numpy.int
    :return: None
    """
    file = h5py.File(path, "a")
    file.create_dataset("data_x", data=data_x)
    file.create_dataset("data_y", data=data_y)
    file.close()


def from_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from hdf5 file.
    :param path: where HDF5 file is stored
    :return: the same as from_directory
    """
    file = h5py.File(path, "r")
    return file["data_x"], file["data_y"]


if __name__ == '__main__':
    x, y = from_directory("data")
    _store_as_hdf5("dataset.h5", x, y)
