from keras.models import Sequential
from keras.layers import Dense, Activation, Softmax
from keras.utils import print_summary


class Perceptron:
    def __init__(self, input_width, input_height, first_hidden_size, second_hidden_size, classes):
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

    def train(self, training_set, validation_set, batch_size, epochs):
        self.model.fit(x=training_set['data'],
                       y=training_set['class_id'],
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True,
                       validation_data=validation_set)
