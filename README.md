# **Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project** 
*My solution to the Udacity Self-Driving Car Engineer Nanodegree Behavioral Cloning project.*

**Note: This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo).**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode(had been modified)
* model.h5 containing a trained convolution neural network 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model modified based on the network described in NVIDA's paper-“End to End Learning for Self-Driving Cars“ , It consists of 5 conv layer and 4 fully connect layer. (model.py lines 201-236)

I use RELU as activation fucntion and add 0.45 drop at each fully connect layer. And the data is normalized in the model using a Keras lambda layer (code line 205).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 224/227/230/233).The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 243). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 240).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to change on mature model so that you can avoid redesigning a model, which is the essence of transfer learning.

So I chose  to use a convolution neural network model similar to the  NVIDA’s model which was introduced in the paper “End to End Learning for Self-Driving Cars“,I thought this model might be appropriate, there are two reasons list below:

First, this project is very similar to NVIDA’s project, and also NVIDA’s model is neither complicated or so large, I even train the model on my own laptop (Due to the network and other factors, GPU performance failed to play well, even often for no reason to interrupt, which wasted too much of my time)

Second, the most important thing is that it had been proved successful, so I can spend more time working outside of the model, such as data balance processing and so on.

I started with three cameras in the center, left, and right three images, and I added the angle correction for the left and right images:

Center_angle = angle

Left_angle = angle + correction

Right_angle = angle – correction

After this I got 8036*3 = 24108 images with angles.

I then processed the image, resizing the image from 160*320 to 66*200, and make sure the processed image was in RGB format. ( model.py line127:136)

And randomly used rotate and shift method to do data augmentation(model.py line143:192). After this I got one time additional data for training and validation.

I split the image and steering angle data into a training and validation set, validation data accounts for 20%.

And then train the model and run the simulator to see how well the car was driving around track one. Unfortunately, the car almost go straight line, it seems that the car has not mastered the skills of turning.

Then I looked at the turning angle values of the data, I found that more than half of the data that the steering angle were zero or close to zero, the model learned from this data that go straight, without learning to turn.

So we need to reduce these smaller angle data sets, I did some experiments and found that if the absolute value of the angle was less than 0.04 or below, and adjust left and right offset angles, In my model car can be driven inside the road without hitting the road boundary.

Finally I chose absolute value of the angle as 0.02, and keep 4% of these data, set the angle offset as 0.21;

Also I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding a drop-layer after each full connect-layer, and set 0.45 as dropped percentages. The final “model MSE loss’’ see bellow pic.

![model MSE loss’]: ./history.PNG

#### 2. Final Model Architecture

The final model architecture (model.py lines 201-244).

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


![alt text][image1]

