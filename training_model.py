from keras.models import Model
import time
import datetime
from data_generator import DataGenerator
import h5py
from model_generator import ModelGenerator, ModelType

# create base model (pre-trained) (VGG16_MODEL, GOOGLENET_MODEL)
category_count = 38
model_type = ModelType.ALEXNET_MODEL

model_generator = ModelGenerator(model_type, category_count)
model, resize_shape = model_generator.getModel()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['acc'])

data_path = "data/plant_disease_classification_data.h5"
file = h5py.File(data_path, 'r')
train_data = file["train_data"]
test_data = file["test_data"]
data_generator = DataGenerator()

print("Train data size: ", len(train_data))
print("Test data size: ", len(test_data))

batch_size = 24
steps_per_epoch = len(train_data) // batch_size
epochs = 15
validation_steps = len(test_data) // batch_size
lr=0.005
momentum=0.9
decay=0.0005

start_time = time.time()
start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')

model.fit_generator(generator=data_generator.generate_data(train_data, resize_shape=resize_shape), steps_per_epoch=steps_per_epoch, epochs=epochs,
                    validation_data=data_generator.generate_data(test_data, resize_shape=resize_shape), validation_steps=validation_steps)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
model = model_generator.prepare_second_model(model)

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay), loss='categorical_crossentropy',  metrics=['acc'])

model.fit_generator(generator=data_generator.generate_data(train_data, resize_shape=resize_shape), steps_per_epoch=steps_per_epoch, epochs=epochs,
                    validation_data=data_generator.generate_data(test_data, resize_shape=resize_shape), validation_steps=validation_steps)

end_time = time.time()
end_time_string = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')
model_filename = 'trained_models/' + str(start_time_string) + "-" + str(end_time_string) + '.h5'

model.save(model_filename)