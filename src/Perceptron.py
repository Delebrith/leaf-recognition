from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import print_summary
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd

import os


class Perceptron:
    def __init__(self, input_width, input_height, first_hidden_size, second_hidden_size, classes):
        self.input_width = input_width
        self.input_height = input_height

        self.model = Sequential([
            Flatten(input_shape=(input_width, input_height, 3)),
            Dense(first_hidden_size, activation='sigmoid'),
            Dense(second_hidden_size, activation='sigmoid'),
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
                                           width_shift_range=15,
                                           height_shift_range=15,
                                           zoom_range=0.1)

        training_set = img_generator.flow_from_dataframe(dataframe=training_frame,
                                                         directory=os.path.join(data_dir, 'training/'),
                                                         x_col='data',
                                                         y_col='class',
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(self.input_width, self.input_height),
                                                         class_mode='categorical')

        validation_set = img_generator.flow_from_dataframe(dataframe=validation_frame,
                                                           directory=os.path.join(data_dir, 'validation/'),
                                                           x_col='data',
                                                           y_col='class',
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(self.input_width, self.input_height),
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
                                            width_shift_range=15,
                                            height_shift_range=15,
                                            zoom_range=0.1)

        test_set = data_generator.flow_from_dataframe(dataframe=test_frame,
                                                      directory=os.path.join(data_dir, 'test/'),
                                                      x_col='data',
                                                      y_col='class',
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      target_size=(self.input_width, self.input_height),
                                                      class_mode='categorical')

        steps_eval = test_set.n // batch_size
        result = self.model.evaluate_generator(generator=test_set,
                                               steps=steps_eval)
        print('Networks score -  loss: %d; accuracy: %d' % (result[0], result[1]))

    def save(self, model_path, history_path):
        self.model.save(model_path, overwrite=True)
        history_frame = pd.DataFrame.from_dict(self.history.history)
        history_frame.to_csv(history_path)

    def load(self, path):
        self.model.load_weights(path)
