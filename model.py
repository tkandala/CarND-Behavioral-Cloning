import numpy as np
import pandas

training_data_file = "driving_log.csv"
dataframe = pandas.read_csv(training_data_file, usecols=['center', 'steering'])
#print(dataframe.values)

X = dataframe.values[:,0]
y = dataframe.values[:,1]
Y = y.reshape((1,) + y.shape)
y = Y[0]

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator()

from PIL import Image

x = []
for i in range(len(X)):
	image = load_img(X[i])
	image_PIL = Image.open(X[i])
	image_array = np.asarray(image_PIL)
	#print(image_array.shape)
	#x = img_to_array(image)
	#x = x.reshape((1,) + x.shape)
	x.append(image_array)

x = np.asarray(x)
X = x[None, :, :, :]
X = X[0]
#X = np.delete(X, 0, 1)

#from sklearn.utils import shuffle
#X,y = shuffle(X,y)

from keras.layers import Dense, Activation, ELU, Convolution2D, Dropout, MaxPooling2D, Lambda, Flatten

def steering_model():
	# create model
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160, 320, 3),output_shape=(160, 320, 3)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	#model.add(ELU())
	#model.add(Convolution2D(80,3,3,border_mode='valid'))
	#model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dropout(0.2))
	#model.add(Activation('relu'))
	model.add(ELU())
	#model.add(Dense(1, init='normal', activation='relu'))
	model.add(Dense(512))
	model.add(ELU())
	model.add(Dense(1))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = steering_model()
history = model.fit(X, y, batch_size=128, nb_epoch=2, validation_split=0.2)

print("Saving model weights and configuration file.")

model_json = model.to_json()

with open("./model.json", "w") as json_file:

    json.dump(model_json, json_file)

model.save_weights("./model.h5")

print("Saved model to disk")



