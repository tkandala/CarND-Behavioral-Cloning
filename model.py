import numpy as np
import pandas
import json
import h5py
import cv2

# read the data form the csv file using pandas
training_data_file = "driving_log.csv"
# Pull in just the information from center camera and steering angle
dataframe = pandas.read_csv(training_data_file, usecols=['center', 'left', 'right', 'steering'])
#print(dataframe.values)

X = dataframe.values[:,0]
X_left = dataframe.values[:,1]
X_right = dataframe.values[:,2]
y = dataframe.values[:,3]

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M, (320, 160))
    
    return image_tr,steer_ang

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#from keras.preprocessing.image import load_img

from PIL import Image

#Prepare Image data for Neural Network
x = []
x_left = []
x_right = []

y_left = []
y_right = []

for i in range(len(X)):
	
	correction = 0.25
	y_left.append(y[i] + correction)
	y_right.append(y[i] - correction)

	#image = load_img(X[i])
	image_PIL = Image.open(X[i])
	image_PIL_left = Image.open(X_left[i])
	image_PIL_right = Image.open(X_right[i])
	
	image_array = np.asarray(image_PIL)
	image_array_left = np.asarray(image_PIL_left)
	image_array_right = np.asarray(image_PIL_right)

	#translation - Horizontal or Vertical shifts
	image_array,y[i] = trans_image(image_array,y[i],100)
	image_array_left,y_left[i] = trans_image(image_array_left,y_left[i],100)
	image_array_right,y_right[i] = trans_image(image_array_right,y_right[i],100)

	#brightness
	image_array = augment_brightness_camera_images(image_array)
	image_array_left = augment_brightness_camera_images(image_array_left)
	image_array_right = augment_brightness_camera_images(image_array_right)

	x.append(image_array)
	x_left.append(image_array_left)
	x_right.append(image_array_right)

x = x + x_left + x_right
x = np.asarray(x)
X = x[None, :, :, :]
X = X[0]

# Prepare steering angle data for Neural Network
y = np.append(y, y_left)
y = np.append(y, y_right)
Y = y.reshape((1,) + y.shape)
y = Y[0]

from keras.layers import Dense, Activation, ELU, Convolution2D, Dropout, MaxPooling2D, Lambda, Flatten, Cropping2D
from keras.optimizers import Adam

def steering_model():
	# create model
	# The model is very similar to https://github.com/commaai/research/blob/master/train_steering_model.py
	# except that a Max-pooling layer has been added after the last Convolutional layer in this network
	# and the second dropout layer has been removed.
	model = Sequential()
	# Crop the images to focus just on the lane
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	# Start off with a Lambda layer that is going to normalize the RGB values between -1 & 1
	model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(90, 320, 3),output_shape=(90, 320, 3)))
	#model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160, 320, 3),output_shape=(160, 320, 3)))
	# Convolutional Layer #1 with depth of 16, 8x8 kernel, 4x4 Stride
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	# ELU Activation Layer #1
	model.add(ELU())
	# Convolutional Layer #2 with depth of 32, 5x5 kernel, 2x2 Stride
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	# ELU Activation Layer #2
	model.add(ELU())
	# Convolutional Layer #3 with depth of 64, 5x5 kernel, 2x2 Stride
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	# ELU Activation Layer #3
	model.add(ELU())
	# Convolutional Layer #4 with depth of 128, 5x5 kernel, 2x2 Stride
	model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode="same"))
	# ELU Activation Layer #4
	model.add(ELU())
	# Convolutional Layer #5 with depth of 256, 5x5 kernel, 2x2 Stride
	model.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode="same"))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2)))
	# Adding a dropout of 40%
	model.add(Dropout(0.4))
	# ELU Activation Layer #5
	model.add(ELU())
	# Flatter layers
	model.add(Flatten())
	# Fully connected layer #1
	model.add(Dense(512))
	# ELU Activation Layer #6
	model.add(ELU())
	# Fully connected layer #2
	model.add(Dense(256))
	# ELU Activation Layer #7
	model.add(ELU())
	# Fully connected layer #3
	model.add(Dense(128))
	# Adding a dropout of 40%
	model.add(Dropout(0.4))
	# ELU Activation Layer #8
	model.add(ELU())
	# Fully connected layer #4 - Will have only 1 output as we are only looking for the Steering angle
	model.add(Dense(1))
	adam = Adam(lr=0.001)
	# Compile model
	model.compile(loss='mean_squared_error', optimizer=adam)
	return model

model = steering_model()

#plot the Keras model
#from keras.utils.visualize_util import plot
#plot(model, to_file='model.png')

# Train the model using the training data using a batch_size of 128 for 5 epochs. Using a validation set of size 20%
history = model.fit(X, y, batch_size=128, nb_epoch=5, validation_split=0.2)

print("Saving model weights and configuration file.")

model_json = model.to_json()

with open("./model.json", "w") as json_file:

    json.dump(model_json, json_file)

model.save_weights("./model_weights.h5")
model.save("./model.h5")

print("Saved model to disk")