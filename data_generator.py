import numpy as np
import os
import cv2
import json
import h5py

from utils.categories_conversion_utils import *
from utils.directory_utils import *


class DataGenerator:

    def __init__(self, images_folder='data/color', batch_size=24):
        self.images_folder = images_folder
        self.batch_size = batch_size
        return

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def save_folder_names_json(self, path_to_json_file="data/category_dictionary.json"):
        category_names = [name for name in os.listdir(self.images_folder)]
        category_dictionary = {}

        category_names = sorted(category_names)
        for i in range(len(category_names)):
            category_dictionary[category_names[i]] = i

        with open(path_to_json_file, 'w') as fp:
            json.dump(category_dictionary, fp)

    def prepare_batch_data(self, data, resize_shape):

        batch_data = np.zeros((len(data), resize_shape[0], resize_shape[1], 3))
        batch_labels = []
        for i in range(len(data)):
            image_path = data[i][0]
            batch_labels.append(category_to_one_hot(data[i][1]))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, resize_shape)
            image = self.preprocess_input(image.astype(float))
            batch_data[i] = image

        return batch_data, np.array(batch_labels)

    def generate_data(self, data, resize_shape, testing=False):

        data = np.array(data)

        while True:

            np.random.shuffle(data)
            num_samples = len(data)

            for i in range(0, num_samples, self.batch_size):
                batch_data = data[i: i + self.batch_size]
                images, labels = self.prepare_batch_data(batch_data, resize_shape)
                yield images, labels

            if testing:
                break