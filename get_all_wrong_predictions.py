import tensorflow as tf  # Default graph is initialized when the library is imported
import os
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import cv2
from data_generator import DataGenerator
from utils.categories_conversion_utils import one_hot_to_category
import h5py

generator = DataGenerator(batch_size=1)
data_file = h5py.File('data/plant_disease_classification_data.h5', 'r')

full_dataset = data_file["train_data"]

with tf.Graph().as_default() as graph:  # Set default graph as graph

    with tf.Session() as sess:
        # Load the graph in graph_def
        print("load graph")

        # We load the protobuf file from the disk and parse it to retrive the unserialized graph_def
        with gfile.FastGFile("trained_models/optimized_mobilenet_plant_graph.pb", 'rb') as f:



            print("Plot image...")
            # scipy.misc.imshow(image)

            # Set FCN graph to the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)

            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )

            # Print the name of operations in the session
            # for op in graph.get_operations():
            #     print "Operation Name :", op.name  # Operation name
            #     print "Tensor Stats :", str(op.values())  # Tensor name

            # INFERENCE Here
            l_input = graph.get_tensor_by_name('input:0')  # Input Tensor
            l_output = graph.get_tensor_by_name('final_result:0')  # Output Tensor

            print "Shape of input : ", tf.shape(l_input)
            # initialize_all_variables
            tf.global_variables_initializer()

            # Run Kitty model on single image
            count = 0
            for batch_images, batch_labels in generator.generate_data(full_dataset,resize_shape=(224,224)):

                predictions = sess.run(l_output, feed_dict={l_input: batch_images})
                print(predictions[0])
                original = one_hot_to_category(batch_labels[0])
                predicted = one_hot_to_category(predictions[0])
                if original != predicted:
                    count += 1
                    print("Original category: ",original)
                    print("Predicted category: ", predicted)


                break
            print("Count: ", count)