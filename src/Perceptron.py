from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import print_summary
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
import numpy as np
from src.NeuralNetwork import NeuralNetwork

import os


class Perceptron(NeuralNetwork):
    def __init__(self, input_width, input_height, first_hidden_size, second_hidden_size, third_hidden_size, classes):
        NeuralNetwork.__init__(self, input_width, input_height, classes)

        self.model = Sequential([
            Flatten(input_shape=(input_height, input_width, 3)),
            Dense(first_hidden_size, activation='sigmoid'),
            Dense(second_hidden_size, activation='sigmoid'),
            Dense(third_hidden_size, activation='sigmoid'),
            Dense(classes, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='Adam',
                           metrics=['accuracy'])
        print_summary(self.model)
        self.history = {}

    def train(self, training_frame, validation_frame, batch_size, epochs, data_dir):
        img_generator = ImageDataGenerator(rescale=1./255.,
                                           rotation_range=15,
                                           horizontal_flip=True,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.1,
                                           zoom_range=0.1)

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
        self.history = self.model.fit_generator(generator=training_set,
                                                steps_per_epoch=steps_training,
                                                validation_data=validation_set,
                                                validation_steps=steps_validation,
                                                epochs=epochs,
                                                workers=8)

    def evaluate(self, test_frame, data_dir, batch_size):
        data_generator = ImageDataGenerator(rescale=1./255.,
                                            rotation_range=15,
                                            horizontal_flip=True,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.1,
                                            zoom_range=0.1)

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
        data_generator = ImageDataGenerator(rescale=1./255.,
                                            rotation_range=15,
                                            horizontal_flip=True,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.1,
                                            zoom_range=0.1)

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
