#**Traffic Sign Recognition** 

##Writeup Report

---

**Behavioral Cloning Project**

### Sample training images

[image1]: ./images/center_2016_12_01_13_30_48_287.jpg "Center camera image"
[image2]: ./images/left_2016_12_01_13_30_48_287.jpg "left camera image"
[image3]: ./images/right_2016_12_01_13_30_48_287.jpg "right camera image"
[image4]: ./images/center_2016_12_01_13_31_15_513.jpg "Center Image near dirt track"
[image5]: ./images/left_2016_12_01_13_31_15_513.jpg "left Image near dirt track"
[image6]: ./images/right_2016_12_01_13_31_15_513.jpg "right Image near dirt track"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of 5 convolution neural network layers with 8x8 & 5x5 filter sizes and depths between 16 and 256 (model.py lines 110-126) 

The model includes multiple RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 108).

A cropping layer (code line 106) (as mentioned inside the lectures) was added to crop off the scenary and the car hood.

Also, a Max-Pooling layer (code line 128), A flatten layer (code line 134) and 4 Dense layers (code lines 136, 140, 144 & 150) have been ended, giving an output of only 1 value which will be the Steering angle prediction.

####2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers with a dropout of 40% in order to reduce overfitting (model.py lines 130 & 146).

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 151). Also a loss function - Mean-Squared Error was used.

####4. Appropriate training data

Probably it is just my computer but I haven't been able to run the simulator at all on Mac (my development environment). I tried to run the simulator with Linux inside Virtualbox but wasn't able to fix the 3D render issue making the simulator utter slow to run. 

I also tried running the simulator in Windows - where fortunately it worked but the training data used there wasn't the best because I used arrow keys to steer the car. This of course is not a smooth steering. Using the beta simulator definitely solves the issue but controlling using a mouse (in absence of a x-box controller or joystick) is an absolute pain. I wasted time trying to properly run the simulator and trying out different versions of the simulator in multiple machines. I ended up using just the sample Udacity training data to train my model. But the sample data in the end worked out to be sufficient to properly train the car.

All three camera images data was used as the training data set. This allowed for additional training data. To compensate for the left and right camera's steering angle, a correction value of 0.25 was used to add/subtract to the center camera steering angle. 

With just the above training data, the following output was produced for 5 epochs

```sh
Train on 12857 samples, validate on 3215 samples
Epoch 1/5
12857/12857 [==============================] - 104s - loss: 0.1007 - val_loss: 0.0137
Epoch 2/5
12857/12857 [==============================] - 103s - loss: 0.0107 - val_loss: 0.0121
Epoch 3/5
12857/12857 [==============================] - 106s - loss: 0.0092 - val_loss: 0.0122
Epoch 4/5
12857/12857 [==============================] - 111s - loss: 0.0081 - val_loss: 0.0116
Epoch 5/5
12857/12857 [==============================] - 110s - loss: 0.0066 - val_loss: 0.0133
```

But we can see here that the loss is too low and validation data loss increased in the 5th epoch meaning that the model was not being trained properly. Of course, the autonomous car went off road near the dirt track - a sharp left turn where there is no right side marker. This meant the training data was not enough or the model was overfitting - even with 2 dropout layers.

With help from my mentor, I researched the data augmentation piece here - https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.roop6bbkv and used the brightness and image translation augmentation strategies mentioned there. Fortunately, after adding these two pieces to the existing training set, the validation set loss become much more relevant as shown below. Although the loss is more than the initial run, we can see that the loss has not increased in 5th epoch and also the training set loss is not too low compared to the original run which means the model is not overfitting.

```sh
Using TensorFlow backend.
Train on 19286 samples, validate on 4822 samples
Epoch 1/5
19286/19286 [==============================] - 174s - loss: 0.3384 - val_loss: 0.1087
Epoch 2/5
19286/19286 [==============================] - 197s - loss: 0.0423 - val_loss: 0.0795
Epoch 3/5
19286/19286 [==============================] - 224s - loss: 0.0354 - val_loss: 0.0844
Epoch 4/5
19286/19286 [==============================] - 243s - loss: 0.0303 - val_loss: 0.0656
Epoch 5/5
19286/19286 [==============================] - 211s - loss: 0.0266 - val_loss: 0.0641
Saving model weights and configuration file.
Saved model to disk
```

Using the new model, the car ran autonomously on the track, including the dirt track area. This can also be viewed in the run1.mp4 video file.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one mentioned here - https://github.com/commaai/research/blob/master/train_steering_model.py but with some changes to the layers itself. I thought this model might be appropriate because it uses the right combination of Convolutional layers with dropout and Dense layers. I added a Max-pooling layer to bring the architecture somewhat similar to the one used for traffic-sign classification although we are solving a complete different scenario here. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set - 20% of training data assigned for validation. I added dropout layers (before even trying to train without one) to avoid overfitting.

As per the lectures, I added a lambda layer to normalize the image color data and a cropping layer to remove the top 50 and bottom 20 pixels from the original image - cutting off the scenary and the car hood parts.

####2. Final Model Architecture

The final model architecture (model.py lines 104-153) consisted of a convolution neural network with the following layers:
5 Convolutional layers, 8 ELU activation function layers, a Max-pooling layer, two Dropout layers, a flatten layer, 4 Dense layers, a lambda layer and a cropping layer.

To start with, the images were cropped top and bottom (50 & 20 pixels respectively) to remove the scenary and car hood. The output of this layer then gets fed into the lambda layer which normalizes the image color data. Next is the combination of Convolution and activation layers. The first Convolution layer starts with a layer depth of 16 (from the input depth of 3), a 8x8 kernel size and 4x4 stride. Next convolutional layer outputs a layer of depth 32 (Stride 2x2) - basically doubling the input layer depth. By the fifth layer, the depth of the layer increases to 256.

A max pooling layer (similar to the Traffic Sign Classification project) was added after the final convolutional layer. Immediately following is a first instance of a dropout layer. Adding a dropout layer reduces overfitting so one layer appears right after the max-pooling layer and the next one before the final fully connected layer. 

Next layer in the architecture is the Flatten layer and then a series of Fully-connected layers that start with and output of 512 until the final (4th) fully-connected layer that has an output of 1 - our steering angle.

I tried to export the model visualtion using the below code but I was not able to properly resolve the required dependencies to visualize the plot. 

```sh
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
```

Although I had pydot and graphviz installed, the error still kept popping for me.

```sh
 raise ImportError('Failed to import pydot. You must install pydot'
ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
```

####3. Creation of the Training Set & Training Process

As metioned earlier, I had issues with simulators and wasn't able to get proper training data on time so I ended up using the sample Udacity training data for my network.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

Initially, the car went off track near the sharp left dirt track area so I tried to train the model with additional training data near the dirt track but that didn't help. I realzed that the dirt track was almost the same color as the asphault track so data augmentation was necessary to properly train the car.

As mentioned earlier, the data augmentation techniques used were referenced from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.roop6bbkv. Two of the many mentioned techniques were used - Brightness and Translation. The brightness augmentation adds the effect of day and night driving and the translation augmentation adds the effect of driving up and down a slope. Adding just the brightness augmentation didn't help successfully driving the car near the dirt track but the translation augmentation immediately fixed that issue. The translation augmentation helped the model understand that the dirt track was to be avoided.