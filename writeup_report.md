#**Traffic Sign Recognition** 

##Writeup Report

---

**Behavioral Cloning Project**

[image7]: ./examples/placeholder_small.png "Flipped Image"

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

My model consists of 5 convolution neural network layers with 8x8 & 5x5 filter sizes and depths between 16 and 256 (model.py lines 67-83) 

The model includes multiple RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 65).

A cropping layer (code line 63) (as mentioned inside the lectures) was added to crop off the scenary and the car hood.

Also, a Max-Pooling layer (code line 85), A flatten layer (code line 87) and 4 Dense layers (code lines 93, 97, 101 & 105) have been ended, giving an output of only 1 value which will be the Steering angle prediction.

####2. Attempts to reduce overfitting in the model

The model contains dropout layer with a dropout of 40% in order to reduce overfitting (model.py lines 89).

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106). Also a loss function - Mean-Squared Error was used.

####4. Appropriate training data

Probably it is just my computer but I haven't been able to run the simulator at all on Mac (my development environment). I tried to run the simulator with Linux inside Virtualbox but wasn't able to fix the 3D render issue making the simulator utter slow to run. 

I also tried running the simulator in Windows - where fortunately it worked but the training data used there wasn't the best because I used arrow keys to steer the car. This of course is not a smooth steering. Using the beta simulator definitely solves the issue but controlling using a mouse (in absence of a x-box controller or joystick) is an absolute pain. I wasted time trying to properly run the simulator and trying out different versions of the simulator in multiple machines. I ended up using just the sample Udacity training data to train my model.

To generalize the training data, the car center image and steering data were flipped and added to the original dataset.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one mentioned here - https://github.com/commaai/research/blob/master/train_steering_model.py but with some changes to the layers itself. I thought this model might be appropriate because it uses the right combination of Convolutional layers with dropout and Dense layers. The network uses 2 drop-out layers but I preferred using only one dropout layer. Also added a Max-pooling layer to bring the architecture somewhat similar to the one used for traffic-sign classification although we are solving a complete different scenario here. Also my network has different number of convolutional layers (3 compared to 5 in mine).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set - 20% of training data assigned for validation. I added dropout layer (before even trying to train without one) to avoid overfitting.

####2. Final Model Architecture

The final model architecture (model.py lines 61-108) consisted of a convolution neural network with the following layers:
5 Convolutional layers, 8 ELU activation function layers, a Max-pooling layer, a Dropout layer, A flatten layer and 4 Dense layers.

####3. Creation of the Training Set & Training Process

As metioned earlier, I had issues with simulators and wasn't able to get proper training data on time so I ended up using the sample Udacity training data for my network.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Below is a output of the training.

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

I used an adam optimizer so that manually training the learning rate wasn't necessary.