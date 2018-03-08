import json
import numpy as np
import os
from glob import glob
import h5py


def category_to_one_hot(folder_name, category_json="data/category_dictionary.json"):
    category_dictionary = json.load(open(category_json))
    one_hot = np.zeros((len(category_dictionary.keys())))
    one_hot[category_dictionary[folder_name]] = 1

    return one_hot


def one_hot_to_category(prediction, category_json="data/category_dictionary.json"):
    category_dictionary = json.load(open(category_json))
    category_index = np.argmax(prediction)

    for category, index in category_dictionary.iteritems():
        if index == category_index:
            return category

    return ""


def get_file_names_from_directory(directory_path, extension):
    return [y for x in os.walk(directory_path) for y in glob(os.path.join(x[0], extension))]


def split_data(images_directory):
    category_names = [name for name in os.listdir(images_directory)]

    train_data = []
    test_data = []

    for category in category_names:
        category_path = images_directory + "/" + category
        images_in_category = get_file_names_from_directory(directory_path=category_path, extension="*.JPG")
        images_in_category = images_in_category + get_file_names_from_directory(directory_path=category_path,
                                                                                extension="*.jpg")

        num_images = len(images_in_category)

        for i in range(num_images):
            if i < int(num_images * 0.8):
                train_data.append([images_in_category[i], category])
            else:
                test_data.append([images_in_category[i], category])

    file = h5py.File('data/plant_disease_classification_data.h5', 'w')
    file.create_dataset("train_data", data=train_data)
    file.create_dataset("test_data", data=test_data)
