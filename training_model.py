from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import time
import datetime
from data_generator import DataGenerator

# create base model (pre-trained)
base_model = InceptionV3(weights='imagenet', include_top=False)

# add layers to train
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(38, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['acc'])

data_generator = DataGenerator()
train_data, test_data = data_generator.split_data()


batch_size = 24
steps_per_epoch = len(train_data) // batch_size
epochs = 30
validation_steps = len(test_data) // batch_size
lr=0.005
momentum=0.9
decay=0.0005

start_time = time.time()
start_time_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H:%M:%S')

model.fit_generator(generator=data_generator.generate_data(train_data), steps_per_epoch=steps_per_epoch, epochs=epochs,
                    validation_data=data_generator.generate_data(test_data), validation_steps=validation_steps)

# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=lr, momentum=momentum, decay=decay), loss='categorical_crossentropy')

model.fit_generator(generator=data_generator.generate_data(train_data), steps_per_epoch=steps_per_epoch, epochs=epochs,
                    validation_data=data_generator.generate_data(test_data), validation_steps=validation_steps)

end_time = time.time()
end_time_string = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d_%H:%M:%S')
model_filename = './trained_models/' + str(start_time_string) + "-" + str(end_time_string) + '.h5'

model.save(model_filename)