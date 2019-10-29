import os

import cv2
import numpy as np

import gesture
import preproc

# Load
resize_width = 120
resize_height = 90


def load_dataset(path: str):
    """
    Load the whole dataset from file and assemble as training data.
    :param path: where to load dataset
    :return: (data_x, data_y)
        data_x.shape = [num_sequences, sequence_len, image_height, image_width, num_channels]
        data_y.shape = [num_sequences]
    """
    data_x = np.ndarray([0, preproc.sequence_length, resize_height, resize_width, 2], dtype=np.float32)
    data_y = np.ndarray([0], dtype=np.int)

    for gesture_id in range(len(gesture.category_names)):
        gesture_dir = os.path.join(path, gesture.category_names[gesture_id])
        if not os.path.exists(gesture_dir):
            continue
        for seq_name in os.listdir(gesture_dir):
            seq_dir = os.path.join(gesture_dir, seq_name)
            seq = load_one_sequence(seq_dir)
            data_x = np.append(data_x, [seq], axis=0)
            data_y = np.append(data_y, gesture_id)

    print("data_x:", data_x.shape, "data_y:", data_y.shape)
    return data_x, data_y


def load_one_sequence(path: str) -> np.ndarray:
    """
    Load one gesture sequence from file and normalize it.
    :param path: where to load a single sequence
    :return: normalized gesture sequence array
        its shape is [sequence_len, image_height, image_width, num_channels]
    """
    # Load depth images and find sequence length
    depth_seq = []
    seq_len = 0
    while True:
        depth_file = os.path.join(path, "d%02d.jpg" % seq_len)
        if not os.path.exists(depth_file):
            break
        depth_image = cv2.imread(depth_file, flags=cv2.IMREAD_UNCHANGED)
        depth_image = cv2.resize(depth_image, (resize_width, resize_height))
        depth_seq.append(depth_image)
        seq_len += 1

    # Load gradient images
    gradient_seq = []
    for i in range(seq_len):
        gradient_file = os.path.join(path, "g%02d.jpg" % i)
        gradient_image = cv2.imread(gradient_file, flags=cv2.IMREAD_UNCHANGED)
        gradient_image = cv2.resize(gradient_image, (resize_width, resize_height))
        gradient_seq.append(gradient_image)

    # Normalize gesture sequence
    normalized = preproc.normalize_sequence([np.array(depth_seq), np.array(gradient_seq)])
    return normalized


if __name__ == '__main__':
    load_dataset("data")
