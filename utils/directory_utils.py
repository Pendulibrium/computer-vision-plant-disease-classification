from glob import glob
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

    for category in category_names:
        category_path = images_directory + "/" + category
        images_in_category_ = get_file_names_from_directory(directory_path=category_path, extensions=["*.JPG", "*.jpg"])
        images_in_category = images_in_category_[0] + images_in_category_[1]

        num_images = len(images_in_category)
        print category
        for i in range(num_images):
            if i < int(num_images * 0.8):
                train_data_images.append(cv2.imread(images_in_category[i]).astype('float'))
                train_data_labels.append(category)
            else:
                test_data_images.append(cv2.imread(images_in_category[i]).astype('float'))
                test_data_labels.append(category)


    file = h5py.File('data/plant_disease_classification_data_images.h5', 'w')
    file.create_dataset("train_data_images", data=train_data_images)
    file.create_dataset("test_data_images", data=test_data_images)
    file.create_dataset("train_data_labels", data=train_data_labels)
    file.create_dataset("test_data_labels", data=test_data_labels)

split_data('/home/wf-admin/Desktop/computer-vision-plant-disease-classification/data/color')