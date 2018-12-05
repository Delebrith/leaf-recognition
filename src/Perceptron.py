from keras.models import Sequential
from keras.layers import Dense, Activation, Softmax
from keras.utils import print_summary
from keras_preprocessing.image import ImageDataGenerator

import os


class Perceptron:
    def __init__(self, input_width, input_height, first_hidden_size, second_hidden_size, classes):
        self.input_width = input_width
        self.input_height = input_height

        self.model = Sequential([
            Dense(first_hidden_size, input_dim=input_width * input_height * 3),
            Activation('sigmoid'),
            Dense(second_hidden_size),
            Activation('sigmoid'),
            Softmax(classes)
        ])

        self.model.compile(loss='mean_squared_error',
                           optimizer='Adam',
                           metrics=['accuracy'])
        print_summary(self.model)

    def train(self, training_frame, validation_frame, batch_size, epochs, data_dir):
        img_generator = ImageDataGenerator(rescale=1./255.)
        training_set = img_generator.flow_from_dataframe(dataframe=training_frame,
                                                         directory=os.path.join(data_dir, 'training/'),
                                                         x_col='data',
                                                         y_col='class',
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(self.input_width, self.input_height))
        # print(training_set.info())

        validation_set = img_generator.flow_from_dataframe(dataframe=validation_frame,
                                                           directory=os.path.join(data_dir, 'validation/'),
                                                           x_col='data',
                                                           y_col='class',
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(self.input_width, self.input_height))
        # print(validation_set.info())
        STEP_SIZE_TRAIN = training_frame.size // batch_size
        STEP_SIZE_VALID = validation_frame.size // batch_size
        self.model.fit_generator(generator=training_set,
                                 steps_per_epoch=STEP_SIZE_TRAIN,
                                 validation_data=validation_set,
                                 validation_steps=STEP_SIZE_VALID,
                                 epochs=epochs,
                                 initial_epoch=1)
