from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Conv2D
from keras_preprocessing.image import ImageDataGenerator
from keras import utils
from keras.optimizers import Adam, RMSprop
from keras import Model
import pandas as pd

import os


def main():
    classes = 32
    input_width = 128
    input_height = 256

    base = VGG16(weights='imagenet', include_top=False, input_shape=(input_height, input_width, 3))

    model = base.output
    model = Flatten(name='flatten')(model)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    output = Dense(classes, activation='softmax', name='predictions')(model)

    model = Model(inputs=base.input, outputs=output)

    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True

    opt = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    utils.print_summary(model)

    augmentation = True
    data_dir = "../../data"
    batch_size = 32
    epochs = 10000

    if augmentation:
        train_generator = ImageDataGenerator(rescale=1.0 / 255.0,
                                             rotation_range=15,
                                             horizontal_flip=True,
                                             width_shift_range=0.15,
                                             height_shift_range=0.5,
                                             zoom_range=0.10,
                                             shear_range=0.10,
                                             fill_mode='nearest')
        val_generator = ImageDataGenerator(rescale=1.0 / 255.0,
                                           rotation_range=15,
                                           horizontal_flip=True,
                                           width_shift_range=0.15,
                                           height_shift_range=0.5,
                                           zoom_range=0.10,
                                           shear_range=0.10,
                                           fill_mode='nearest')
    else:
        train_generator = ImageDataGenerator(rescale=1. / 255.)
        val_generator = ImageDataGenerator(rescale=1. / 255.)

    training_set = train_generator.flow_from_directory(directory=os.path.join(data_dir, 'sets', 'training/'),
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       target_size=(input_height, input_width),
                                                       class_mode='categorical')

    validation_set = val_generator.flow_from_directory(directory=os.path.join(data_dir, 'sets', 'validation/'),
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       target_size=(input_height, input_width),
                                                       class_mode='categorical')

    steps_training = training_set.n // batch_size
    steps_validation = validation_set.n // batch_size

    early_stopping = EarlyStopping(min_delta=0.01, patience=5, restore_best_weights=True)

    history = model.fit_generator(generator=training_set,
                                  steps_per_epoch=steps_training,
                                  validation_data=validation_set,
                                  validation_steps=steps_validation,
                                  epochs=epochs,
                                  workers=8,
                                  callbacks=[early_stopping])


    data_generator = ImageDataGenerator(rescale=1. / 255.)

    test_set = data_generator.flow_from_directory(directory=os.path.join(data_dir, 'sets', 'test/'),
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  target_size=(input_height, input_width),
                                                  class_mode='categorical')

    steps_eval = test_set.n // batch_size
    result = model.evaluate_generator(generator=test_set,
                                      steps=steps_eval)
    print('Networks score -  loss: {}; accuracy: {}'.format(result[0], result[1]))

    model.save(os.path.join(data_dir, "transfer/vgg16.hdf5"), overwrite=True)
    history_frame = pd.DataFrame.from_dict(history.history)
    history_frame.to_csv(os.path.join(data_dir, "transfer/vgg16-history.csv"))


if __name__ == '__main__':
    main()
