from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import print_summary
import pandas as pd
import numpy as np

import os


class DfPerceptron:
    def __init__(self, first_hidden_size, second_hidden_size, third_hidden_size, classes):
        self.input_width = 6
        self.input_height = 1000
        self.classes = classes

        self.model = Sequential([
            Flatten(input_shape=(self.input_height, self.input_width)),
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
        training_x = []
        training_y = []
        for record in training_frame.iterrows():
            image = pd.read_csv(os.path.join(data_dir + '/Folio', record[1]['file']),
                                usecols=['angle', 'octave', 'x', 'y', 'response', 'size']).values
            category = np.array(self.to_categorical(record[1]['class_id']))
            training_x = np.append(arr=training_x, values=image)
            training_y = np.append(arr=training_y, values=category)

        training_x = training_x.reshape((training_frame.size // 3, 1000, 6))
        training_y = training_y.reshape((training_frame.size // 3, self.classes))

        validation_x = []
        validation_y = []
        for record in validation_frame.iterrows():
            image = pd.read_csv(os.path.join(data_dir + '/Folio', record[1]['file']),
                                usecols=['angle', 'octave', 'x', 'y', 'response', 'size']).values
            category = np.array(self.to_categorical(record[1]['class_id']))
            validation_x = np.append(arr=validation_x, values=image)
            validation_y = np.append(arr=validation_y, values=category)

        validation_x = validation_x.reshape((validation_frame.size // 3, 1000, 6))
        validation_y = validation_y.reshape((validation_frame.size // 3, self.classes))

        self.history = self.model.fit(x=training_x,
                                      y=training_y,
                                      validation_data=(validation_x, validation_y),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      epochs=epochs)

    def evaluate(self, test_frame, data_dir, batch_size):
        test_x = []
        test_y = []
        for record in test_frame.iterrows():
            image = pd.read_csv(os.path.join(data_dir + '/Folio', record[1]['file']),
                                usecols=['angle', 'octave', 'x', 'y', 'response', 'size']).values
            category = np.array(self.to_categorical(record[1]['class_id']))
            test_x = np.append(arr=test_x, values=image)
            test_y = np.append(arr=test_y, values=category)

        test_x = test_x.reshape((test_frame.size // 3, 1000, 6))
        test_y = test_y.reshape((test_frame.size // 3, self.classes))

        result = self.model.evaluate(x=test_x,
                                     y=test_y,
                                     batch_size=batch_size)
        print('Networks score -  loss: {}; accuracy: {}'.format(result[0], result[1]))

    def save(self, model_path, history_path):
        self.model.save(model_path, overwrite=True)
        history_frame = pd.DataFrame.from_dict(self.history.history)
        history_frame.to_csv(history_path)

    def load(self, path):
        self.model.load_weights(path)

    def to_categorical(self, class_id):
        array = np.zeros(self.classes).tolist()
        array[int(class_id)] = 1.0
        return array

