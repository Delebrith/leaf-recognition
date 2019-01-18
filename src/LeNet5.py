from src.NeuralNetwork import NeuralNetwork
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from keras_preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal, RandomUniform, glorot_normal, zeros, glorot_uniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD
from keras.datasets import cifar10
from keras.metrics import top_k_categorical_accuracy
from keras import utils
from sklearn.metrics import roc_curve
import numpy as np
import cv2

import os


class LeNet5(NeuralNetwork):
    def __init__(self, input_width, input_height, classes, filter_size, filters, regularization, learning_rate):
        NeuralNetwork.__init__(self, input_width, input_height, classes)

        self.model = Sequential([
            InputLayer(input_shape=(input_height, input_width, 3)),

            Conv2D(data_format='channels_last', filters=filters, kernel_size=(filter_size, filter_size), padding='same',
                   kernel_regularizer=regularization, activation='relu'),


            MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'),

            Conv2D(data_format='channels_last', filters=2*filters, kernel_size=(filter_size, filter_size),
                   padding='same', kernel_regularizer=regularization, activation='relu'),

            MaxPooling2D(pool_size=2, strides=2, data_format='channels_last'),

            Flatten(),

            Dense(120, activation='relu', kernel_regularizer=regularization),

            Dense(84, activation='relu', kernel_regularizer=regularization),

            Dense(classes, activation='softmax')
        ])

        adam = Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])
        utils.print_summary(self.model)
        self.history = {}

    def train(self, batch_size, epochs, data_dir, augmentation=False):
        if augmentation:
            train_generator = ImageDataGenerator(rescale=1. / 255.,
                                                 rotation_range=15,
                                                 horizontal_flip=True,
                                                 width_shift_range=0.15,
                                                 height_shift_range=0.05,
                                                 zoom_range=0.10,
                                                 shear_range=0.10,
                                                 fill_mode='nearest')
            val_generator = ImageDataGenerator(rescale=1. / 255.,
                                               rotation_range=15,
                                               horizontal_flip=True,
                                               width_shift_range=0.15,
                                               height_shift_range=0.05,
                                               zoom_range=0.10,
                                               shear_range=0.10,
                                               fill_mode='nearest')
        else:
            train_generator = ImageDataGenerator(rescale=1./255.)
            val_generator = ImageDataGenerator(rescale=1./255.)

        training_set = train_generator.flow_from_directory(directory=os.path.join(data_dir, 'training/'),
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(self.input_height, self.input_width),
                                                           class_mode='categorical')

        validation_set = val_generator.flow_from_directory(directory=os.path.join(data_dir, 'validation/'),
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(self.input_height, self.input_width),
                                                           class_mode='categorical')

        steps_training = training_set.n // batch_size
        steps_validation = validation_set.n // batch_size

        early_stopping = EarlyStopping(min_delta=0.01, patience=5, restore_best_weights=True)

        self.history = self.model.fit_generator(generator=training_set,
                                                steps_per_epoch=steps_training,
                                                validation_data=validation_set,
                                                validation_steps=steps_validation,
                                                epochs=epochs,
                                                workers=8,
                                                callbacks=[early_stopping])

    def evaluate(self, data_dir, batch_size):
        data_generator = ImageDataGenerator(rescale=1./255.)

        test_set = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'test/'),
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      target_size=(self.input_height, self.input_width),
                                                      class_mode='categorical')

        steps_eval = test_set.n // batch_size
        result = self.model.evaluate_generator(generator=test_set,
                                               steps=steps_eval)
        print('Networks score -  loss: {}; accuracy: {}'.format(result[0], result[1]))

        # batch = test_set.next()
        # for img, cl in zip(batch[0], batch[1]):
        #     print(str(self.model.predict_classes([[img]], batch_size=1)))
        #     print(str(cl))

    def draw_roc(self, data_dir):
        data_generator = ImageDataGenerator(rescale=1./255.)

        test_set = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'test/'),
                                                      batch_size=32,
                                                      shuffle=False,
                                                      target_size=(self.input_height, self.input_width),
                                                      class_mode='categorical')
        steps_eval = test_set.n // 32
        predicted_y = self.model.predict_generator(generator=test_set,
                                                   steps=steps_eval).ravel()
        test_set.reset()
        test_y = test_set.next()[1]
        for _ in range(steps_eval - 1):
            test_y = np.append(arr=test_y, values=test_set.next()[1])
        fpr, tpr, _ = roc_curve(test_y, predicted_y)
        return fpr, tpr

    def train_cifar(self, batch_size, epochs, augmentation=False):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = utils.to_categorical(y_train, self.classes)
        y_test = utils.to_categorical(y_test, self.classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        if augmentation:
            generator = ImageDataGenerator(rescale=1.0 / 255.0,
                                           rotation_range=15,
                                           horizontal_flip=True,
                                           width_shift_range=0.15,
                                           height_shift_range=0.5,
                                           zoom_range=0.10,
                                           shear_range=0.10,
                                           fill_mode='nearest')
            generator.fit(x_train)

            early_stopping = EarlyStopping(min_delta=0.01, patience=5, restore_best_weights=True)
            self.history = self.model.fit_generator(generator.flow(x_train, y_train),
                                                    callbacks=[early_stopping],
                                                    validation_data=(x_test, y_test),
                                                    epochs=epochs,
                                                    workers=8,
                                                    steps_per_epoch=len(x_train) // batch_size,
                                                    validation_steps=len(x_test) // batch_size)
        else:
            self.history = self.model.fit(x_train, y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_data=(x_test, y_test),
                                          shuffle=True)

    def top5(self, data_dir, batch_size):
        data_generator = ImageDataGenerator(rescale=1./255.)

        test_set = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'test/'),
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      target_size=(self.input_height, self.input_width),
                                                      class_mode='categorical')
        steps_eval = test_set.n // batch_size
        predicted_y = self.model.predict_generator(generator=test_set,
                                                   steps=steps_eval)
        test_set.reset()
        test_y = test_set.next()[1]
        for i in range(steps_eval - 1):
            test_y = np.append(arr=test_y, values=test_set.next()[1]).reshape(((i+2) * batch_size, self.classes))
        top5 = top_k_categorical_accuracy(test_y, predicted_y, 5)
        print("Top-5: {}", top5)

    def predict(self, image):
        image = cv2.resize(image, dsize=(self.input_width, self.input_height))
        image_array = np.asarray(image, dtype=float)
        image_array *= 1./255.
        return self.model.predict(x=[[image_array]], batch_size=1)

    def pop_last_layers(self):
        self.model.pop()
        self.model.pop()
