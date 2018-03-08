from keras import Model
from keras.models import load_model
from data_generator import DataGenerator
import util as util


model_name = "2018-03-07_20:51:35-2018-03-07_23:02:18"
model_path = "trained_models/"+model_name+".h5"
model = load_model(model_path)

generator = DataGenerator()
_, test_data = generator.split_data()
i = 0
for batch_images, batch_labels  in generator.generate_data(test_data):
    predictions = model.predict(batch_images)
    #print(predictions)

    for i, prediction in enumerate(predictions):
        print("Original category: ", util.one_hot_to_category(batch_labels[i]))
        print("Predicted category: ", util.one_hot_to_category(prediction))

    break
