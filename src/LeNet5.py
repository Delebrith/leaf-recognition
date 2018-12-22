from src.NeuralNetwork import NeuralNetwork
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from keras.utils import print_summary
from keras_preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from keras.initializers import RandomNormal, RandomUniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import roc_curve
import numpy as np

import os


class LeNet5(NeuralNetwork):
    def __init__(self, input_width, input_height, classes, filter_size, filters, regularization, learning_rate):
        NeuralNetwork.__init__(self, input_width, input_height, classes)
        initializer = RandomUniform(seed=1)

        self.model = Sequential([
            InputLayer(input_shape=(input_height, input_width, 3)),

            Conv2D(data_format='channels_last', filters=filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu',
                   kernel_initializer=initializer, bias_initializer=initializer),


            MaxPooling2D(pool_size=(2, 2), data_format='channels_last'),

            Conv2D(data_format='channels_last', filters=filters, kernel_size=2*filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu',
                   kernel_initializer=initializer, bias_initializer=initializer),

            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_last'),

            Flatten(),

            Dense(512, activation='relu', kernel_regularizer=regularization,
                  kernel_initializer=initializer, bias_initializer=initializer),

            Dense(classes, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer)
        ])

        adam = Adam(lr=learning_rate)
        rmsprop = RMSprop(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])
        print_summary(self.model)
        self.history = {}

    def train(self, training_frame, validation_frame, batch_size, epochs, data_dir, augmentation=False):
        if augmentation:
            img_generator = ImageDataGenerator(rescale=1./255.,
                                               rotation_range=15,
                                               horizontal_flip=True,
                                               width_shift_range=0.15,
                                               height_shift_range=0.15,
                                               shear_range=10,
                                               zoom_range=0.1)
        else:
            img_generator = ImageDataGenerator(rescale=1./255.)

        training_set = img_generator.flow_from_dataframe(dataframe=training_frame,
                                                         directory=os.path.join(data_dir, 'training/'),
                                                         x_col='data',
                                                         y_col='class',
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(self.input_height, self.input_width),
                                                         class_mode='categorical')

        validation_set = img_generator.flow_from_dataframe(dataframe=validation_frame,
                                                           directory=os.path.join(data_dir, 'validation/'),
                                                           x_col='data',
                                                           y_col='class',
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

    def evaluate(self, test_frame, data_dir, batch_size):
        data_generator = ImageDataGenerator(rescale=1./255.)

        test_set = data_generator.flow_from_dataframe(dataframe=test_frame,
                                                      directory=os.path.join(data_dir, 'test/'),
                                                      x_col='data',
                                                      y_col='class',
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      target_size=(self.input_height, self.input_width),
                                                      class_mode='categorical')

        steps_eval = test_set.n // batch_size
        result = self.model.evaluate_generator(generator=test_set,
                                               steps=steps_eval)
        print('Networks score -  loss: {}; accuracy: {}'.format(result[0], result[1]))

    def draw_roc(self, test_frame, data_dir):
        data_generator = ImageDataGenerator(rescale=1./255.)

        test_set = data_generator.flow_from_dataframe(dataframe=test_frame,
                                                      directory=os.path.join(data_dir, 'test/'),
                                                      x_col='data',
                                                      y_col='class',
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

    def top5(self, test_frame, data_dir, batch_size):
        data_generator = ImageDataGenerator(rescale=1./255.)

        test_set = data_generator.flow_from_dataframe(dataframe=test_frame,
                                                      directory=os.path.join(data_dir, 'test/'),
                                                      x_col='data',
                                                      y_col='class',
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
