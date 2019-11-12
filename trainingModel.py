import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, Cropping2D

from keras.callbacks import ModelCheckpoint, TensorBoard

analysis = False


# Load data
def load_data(file_name):
    lines = []
    angles = []
    angles_filtered = []
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            angle = float(row[3])
            angles.append(angle)

            if angle == 0 and np.random.uniform() < 0.9:
                continue

            angles_filtered.append(angle)
            lines.append(row)
            if analysis:
                angle_data_total = np.array(angles)
                plt.figure()
                plt.hist(angle_data_total, bins=40, normed=1)
                plt.xlabel("Steering Angle")
                plt.ylabel("Frequency")
                plt.savefig("images/dataset_full.png")
                angle_data_filtered = np.array(angles_filtered)
                plt.figure()
                plt.hist(angle_data_filtered, bins=40, normed=1)
                plt.xlabel("Steering Angle")
                plt.ylabel("Frequency")
                plt.savefig("images/dataset_preprocessed.png")

    return lines


def generator(lines, batch_size=32, steering_correction=0.3):
    num_samples = len(lines)
    correction = [0, steering_correction, -steering_correction]
    while 1:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset + batch_size]
            images, angles = [], []
            for batch_sample in batch_samples:
                # Loop for 3 cameras
                for i in range(3):
                    image_name = os.path.join("./IMG/", os.path.split(batch_sample[i])[-1])
                    image = mpimg.imread(image_name)
                    angle = float(batch_sample[3]) + correction[i]
                    images.append(image)
                    angles.append(angle)

                    fliped_image = np.fliplr(image)
                    fliped_angle = -angle
                    images.append(fliped_image)
                    angles.append(fliped_angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


def model_define():
    # Model
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model


def main():
    batch_size = 256
    epochs = 50

    samples = load_data('driving_log.csv')

    training_set, validation_set = train_test_split(samples, test_size=0.2)

    trainging_generator = generator(training_set)

    validation_generator = generator(validation_set)

    model = model_define()

    model.compile(loss='mse', optimizer='adam')

    checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)

    callback_list = [checkpoint, TensorBoard(log_dir="./log")]

    model.fit_generator(trainging_generator, steps_per_epoch=len(training_set) / batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(validation_set) / batch_size, epochs=epochs, verbose=1,
                        callbacks=callback_list)


if __name__ == "__main__":
    main()
