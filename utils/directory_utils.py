from glob import glob
import numpy as np
import os
import h5py
import cv2


def get_file_names_from_directory(directory_path, extensions):
    results = []
    for extension in extensions:
        results.append([y for x in os.walk(directory_path) for y in glob(os.path.join(x[0], extension))])
    return results


def split_data(images_directory):
    category_names = [name for name in os.listdir(images_directory)]
    train_data_images = []
    train_data_labels = []
    test_data_images = []
    test_data_labels = []

    folder_index = 0

    for category in category_names:
        category_path = images_directory + "/" + category
        images_in_category_ = get_file_names_from_directory(directory_path=category_path, extensions=["*.JPG", "*.jpg"])
        images_in_category = images_in_category_[0] + images_in_category_[1]
        num_images = len(images_in_category)

        print ("Folder:", category)
        for i in range(num_images):
            if i < int(num_images * 0.8):
                train_data_images.append(cv2.imread(images_in_category[i]).astype('float'))
                train_data_labels.append(category)
            else:
                test_data_images.append(cv2.imread(images_in_category[i]).astype('float'))
                test_data_labels.append(category)

        if num_images > 0:
            file = h5py.File('../data/plant_disease_classification_' + str(folder_index)+ '.h5', 'w')
            file.create_dataset("train_data_images", data=train_data_images)
            file.create_dataset("test_data_images", data=test_data_images)
            file.create_dataset("train_data_labels", data=train_data_labels)
            file.create_dataset("test_data_labels", data=test_data_labels)
            folder_index += 1

#split_data("/Users/wf-markosmilevski/computer-vision-plant-disease-classification/data/color")


def generate_different_background_images(backgrounds_path="../data/backgrounds/",
                                         images_directory="../data/segmented/"):
    category_names = [name for name in os.listdir(images_directory)]
    background_image_files = get_file_names_from_directory(directory_path=backgrounds_path, extensions=["*.jpg"])

    num_background_filters = len(background_image_files[0])
    for i in range(num_background_filters):
        current_background = cv2.imread(background_image_files[i])[:256, :256, :]
        background_image_name_dir = background_image_files[i].rsplit( "/", 1 )[ -1 ].rsplit(".", 1)[0]

        path = '../data/' + background_image_name_dir

        if not os.path.exists(path):
            os.makedirs(path)
        for category in category_names:
            category_path = images_directory + "/" + category
            images_in_category_ = get_file_names_from_directory(directory_path=category_path,
                                                                extensions=["*.JPG", "*.jpg"])
            images_in_category = images_in_category_[0] + images_in_category_[1]
            num_images = len(images_in_category)

            image_save_path = path + '/' + category

            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)

            for i in range(num_images):
                current_image = cv2.imread(images_in_category[i])
                current_image = cv2.resize(current_image, (256, 256))

                current_image_name = images_in_category[i].rsplit("/", 1)[-1]
                current_inverted = invert_image(current_image)

                transformed_image = current_inverted * current_background + current_image
                cv2.imwrite(image_save_path + '/' + current_image_name, transformed_image)

def invert_image(image):
    image_copy = image.copy()
    image_copy[np.where((image_copy == [0, 0, 0]).all(axis=2))] = 1
    image_copy[np.where((image_copy != [1, 1, 1]).all(axis=2))] = 0
    return image_copy