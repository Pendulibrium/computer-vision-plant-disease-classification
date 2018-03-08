import json
import numpy as np


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
