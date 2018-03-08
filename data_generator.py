import numpy as np
import os
import cv2
from glob import glob
import json
import util as util


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

    def save_folder_names_json(self, path_to_json_file="data/folder_dictionary.json"):
        category_names = [name for name in os.listdir(self.images_folder)]
        category_dictionary = {}

        for i in range(len(category_names)):
            category_dictionary[category_names[i]]=i

        with open(path_to_json_file, 'w') as fp:
            json.dump(category_dictionary, fp)

    def split_data(self):
        folder_names = [name for name in os.listdir(self.images_folder)]
        images_paths_JPG = [y for x in os.walk('data/color') for y in glob(os.path.join(x[0], "*.JPG"))]
        images_paths_jpg = [y for x in os.walk('data/color') for y in glob(os.path.join(x[0], "*.jpg"))]

        images_paths = images_paths_JPG + images_paths_jpg
        data = []

        for image_path in images_paths:
            folder_name = ""
            for folder in folder_names:
                if folder in image_path:
                    folder_name = folder
                    break

            data.append([image_path, folder_name])

        data = np.array(data)
        np.random.shuffle(data)
        data_length = len(data)
        split_index = int(data_length * 0.8)
        train_data = data[0:split_index]
        test_data = data[split_index:data_length]

        return train_data, test_data


    def prepare_batch_data(self, data):

        batch_data = np.zeros((len(data), 299, 299, 3))
        batch_labels = []
        for i in range(len(data)):
            image_path = data[i][0]
            batch_labels.append(util.category_to_one_hot(data[i][1]))
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (299,299))
            image = self.preprocess_input(image.astype(float))
            batch_data[i] = image

        return batch_data, np.array(batch_labels)

    def generate_data(self, data, testing=False):

        while True:
            num_samples = len(data)

            for i in range(0, num_samples, self.batch_size):
                batch_data = data[i: i + self.batch_size]
                images, labels = self.prepare_batch_data(batch_data)
                yield [images], labels

            if testing:
                break
