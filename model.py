import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []
images = []
measurements = []

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

##Loop through each line of the file and get all three images, steering wheel angles
for line in lines:
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('/')
		filename = tokens[-1]
		local_path = "./data/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
	correction = 0.1
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement + correction)
	measurements.append(measurement -  correction)

##flip the images and add the copies
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

##lenet architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')


