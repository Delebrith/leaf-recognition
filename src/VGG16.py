from src.NeuralNetwork import NeuralNetwork
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from keras.utils import print_summary
from keras_preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from keras.initializers import RandomNormal, RandomUniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.metrics import roc_curve
import numpy as np

import os


class VGG16(NeuralNetwork):
    def __init__(self, input_width, input_height, classes, filter_size, filters, regularization, learning_rate):
        NeuralNetwork.__init__(self, input_width, input_height, classes)
        initializer = RandomNormal(seed=1, mean=0.0)

        self.model = Sequential([
            InputLayer(input_shape=(input_height, input_width, 3)),
            # block 1
            Conv2D(data_format='channels_last', filters=filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_1_1',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_1_2',
                   kernel_initializer=initializer, bias_initializer=initializer),
            MaxPooling2D(name='block_1_polling', pool_size=(2, 2), data_format='channels_last', strides=(2, 2)),

            # block 2
            Conv2D(data_format='channels_last', filters=2*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_2_1',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=2*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_2_2',
                   kernel_initializer=initializer, bias_initializer=initializer),
            MaxPooling2D(name='block_2_polling', pool_size=(2, 2), data_format='channels_last', strides=(2, 2)),

            # block 3
            Conv2D(data_format='channels_last', filters=4*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_3_1',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=4*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_3_2',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=4*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_3_3',
                   kernel_initializer=initializer, bias_initializer=initializer),
            MaxPooling2D(name='block_3_polling', pool_size=(2, 2), data_format='channels_last', strides=(2, 2)),

            # block 4
            Conv2D(data_format='channels_last', filters=8*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_4_1',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=8*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_4_2',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=8*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_4_3',
                   kernel_initializer=initializer, bias_initializer=initializer),
            MaxPooling2D(name='block_4_polling', pool_size=(2, 2), data_format='channels_last', strides=(2, 2)),

            # block 5
            Conv2D(data_format='channels_last', filters=8*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_5_1',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=8*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_5_2',
                   kernel_initializer=initializer, bias_initializer=initializer),
            Conv2D(data_format='channels_last', filters=8*filters, kernel_size=filter_size, padding='same',
                   kernel_regularizer=regularization, activation='relu', name='block_5_3',
                   kernel_initializer=initializer, bias_initializer=initializer),
            MaxPooling2D(name='block_5_polling', pool_size=(2, 2), data_format='channels_last', strides=(2, 2)),

            # fully_concatenated
            Flatten(),
            Dense(4096, activation='relu', kernel_regularizer=regularization,
                  kernel_initializer=initializer, bias_initializer=initializer),
            Dense(4096, activation='relu', kernel_regularizer=regularization,
                  kernel_initializer=initializer, bias_initializer=initializer),
            Dense(classes, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer)
        ])

        adam = Adam(lr=learning_rate)
        rmsprop = RMSprop(lr=learning_rate)
        sgd = SGD(lr=learning_rate, momentum=0.9)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        print_summary(self.model)
        self.history = {}

    def train(self, training_frame, validation_frame, batch_size, epochs, data_dir, augmentation=False):
        if augmentation:
            train_generator = ImageDataGenerator(rescale=1. / 255.,
                                                 rotation_range=15,
                                                 horizontal_flip=True,
                                                 width_shift_range=15.0,
                                                 height_shift_range=10.0,
                                                 zoom_range=0.10,
                                                 shear_range=0.10,
                                                 fill_mode='nearest')
            val_generator = ImageDataGenerator(rescale=1. / 255.,
                                               rotation_range=15,
                                               horizontal_flip=True,
                                               width_shift_range=15.0,
                                               height_shift_range=10.0,
                                               zoom_range=0.10,
                                               shear_range=0.10,
                                               fill_mode='nearest')
        else:
            train_generator = ImageDataGenerator(rescale=1./255.)
            val_generator = ImageDataGenerator(rescale=1./255.)

        training_set = train_generator.flow_from_dataframe(dataframe=training_frame,
                                                           directory=os.path.join(data_dir, 'training/'),
                                                           x_col='data',
                                                           y_col='class',
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(self.input_height, self.input_width),
                                                           class_mode='categorical')

        validation_set = val_generator.flow_from_dataframe(dataframe=validation_frame,
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
