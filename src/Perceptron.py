from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import print_summary
from keras_preprocessing.image import ImageDataGenerator

import os
import pickle


class Perceptron:
    def __init__(self, input_width, input_height, first_hidden_size, second_hidden_size, classes):
        self.input_width = input_width
        self.input_height = input_height

        self.model = Sequential([
            Flatten(input_shape=(input_width, input_height, 3)),
            Dense(first_hidden_size, activation='sigmoid', input_shape=(input_width, input_height, 3)),
            Dense(second_hidden_size, activation='sigmoid'),
            Dense(classes, activation='softmax')
        ])

        self.model.compile(loss='mean_squared_error',
                           optimizer='Adam',
                           metrics=['accuracy'])
        print_summary(self.model)
        self.history = {}

    def train(self, training_frame, validation_frame, batch_size, epochs, data_dir):
        img_generator = ImageDataGenerator(rescale=1./255.)
        training_set = img_generator.flow_from_dataframe(dataframe=training_frame,
                                                         directory=os.path.join(data_dir, 'training/'),
                                                         x_col='data',
                                                         y_col='class',
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(self.input_width, self.input_height),
                                                         class_mode='sparse')

        validation_set = img_generator.flow_from_dataframe(dataframe=validation_frame,
                                                           directory=os.path.join(data_dir, 'validation/'),
                                                           x_col='data',
                                                           y_col='class',
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(self.input_width, self.input_height),
                                                           class_mode='sparse')

        step_size_training = training_set.n // batch_size
        step_size_validation = validation_set.n // batch_size
        self.history = self.model.fit_generator(generator=training_set,
                                                steps_per_epoch=step_size_training,
                                                validation_data=validation_set,
                                                validation_steps=step_size_validation,
                                                epochs=epochs,
                                                workers=8)
    def save(self, model_path, history_path):
        self.model.save(model_path, overwrite=True)
        with open(os.path.join(history_path, 'history.pickle'), 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)


    def load(self, path):
        self.model.load_weights(path)
