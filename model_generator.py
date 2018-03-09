from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, merge, ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.models import Model

# Return model based on model_name parameter (VGG16_MODEL, GOOGLENET_MODEL, ALEXNET_MODEL)


class ModelGenerator:

    def __init__(self, model_name, number_of_classes):
        self.model_name = model_name
        self.number_of_classes = number_of_classes

    def getModel(self):
        if self.model_name == ModelType.VGG16_MODEL:
            self.current_model = VGG16Model(self.number_of_classes).model()
            return self.current_model, (224, 224)
        if self.model_name == ModelType.GOOGLENET_MODEL:
            self.current_model = GoogLeNetModel(self.number_of_classes).model()
            return self.current_model, (299, 299)
        if self.model_name == ModelType.ALEXNET_MODEL:
            self.current_model = AlexNetModel(self.number_of_classes, input_shape=(256, 256, 3)).model()
            return self.current_model, (256, 256)

    def prepare_second_model(self, model):
        if self.model_name == ModelType.ALEXNET_MODEL:
            return model
        else:
            for layer in model.layers[:249]:
                layer.trainable = False
            for layer in model.layers[249:]:
                layer.trainable = True
            return model


# class for creating pre-trained vgg-16 model with additional FC layer (1024 neurons) and softmax layer with
# number_of_classes categories
class VGG16Model():

    def __init__(self, number_of_classes, weights='imagenet', include_top=False):
        self.weights = weights
        self.include_top = include_top
        self.number_of_classes = number_of_classes

    def model(self):
        # create base model vgg16 (pre-trained)
        base_model = VGG16(weights=self.weights, include_top=self.include_top)
        # add additional layers and set number of outputs
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add another FC layer with 1024 neurons
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer with number_of_classes
        predictions = Dense(self.number_of_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        return model


# class for pre-trained googlenet (v3) model with additional FC layer (1024 neurons) and softmax layer with
# number_of_classes categories
class GoogLeNetModel:

    def __init__(self, number_of_classes, weights='imagenet', include_top=False):
        self.weights = weights
        self.include_top = include_top
        self.number_of_classes = number_of_classes

    def model(self):
        # create base model googlenet v3
        base_model = InceptionV3(weights=self.weights, include_top=self.include_top)

        # add additional layers to the base model and set outputs according to number of categories
        x = base_model.output
        # add globalAveragePooling ()
        x = GlobalAveragePooling2D()(x)
        # add another FC layer with 1024 neurons
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer (number_of_classes classes)
        predictions = Dense(self.number_of_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        self.current_model = model

        for layer in base_model.layers:
            layer.trainable = False

        return model


class AlexNetModel:

    def __init__(self, number_of_classes, input_shape):
        self.number_of_classes = number_of_classes
        self.input_shape = input_shape

    def model(self):
        input = Input(self.input_shape)

        # CONV1 -> MAXPOOL1 -> NORM1
        x = Conv2D(96, (11, 11), strides=(4, 4), name='conv1')(input)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool1')(x)
        x = BatchNormalization(axis=3, name='norm1')(x)
        x = ZeroPadding2D((2, 2))(x)

        # CONV2 -> MAXPOOL2 -> NORM2
        x = Conv2D(256, (5, 5), strides=(1, 1), name='conv2', padding='valid')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool2', padding='valid')(x)
        x = BatchNormalization(axis=3, name='norm2')(x)
        x = ZeroPadding2D((1, 1))(x)

        # CONV3 -> CONV4 -> CONV5 -> MAXPOOL3
        x = Conv2D(384, (3, 3), strides=(1, 1), name='conv3')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), name='conv4')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), name='conv5')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='maxpool3')(x)
        x = ZeroPadding2D((1, 1))(x)

        # FC
        x = Flatten()(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        predictions = Dense(self.number_of_classes, activation='softmax', name='fc3')(x)

        model = Model(inputs=input, outputs=predictions)

        print (type(model))
        return model

# enum class for all possible models
from enum import Enum
class ModelType(Enum):
    GOOGLENET_MODEL = "googleNet"
    VGG16_MODEL = "vgg16"
    ALEXNET_MODEL = "alexNet"