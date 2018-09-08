# **Behavioral Cloning Project** 
---

## Purpose

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. 

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. 

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

[//]: # (Image References)

[image1]: ./examples/center_2016_12_01_13_30_48_287.jpg
[image2]: ./examples/left_2016_12_01_13_30_48_287.jpg
[image3]: ./examples/right_2016_12_01_13_30_48_287.jpg
[image4]: ./examples/left_flip.jpg
[image5]: ./examples/MSE.png
[image6]: ./examples/ModelStructure.JPG 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Besides of 1 cropping layer, 1 lambda layer and 1 output layers, the final architecture included 5 convolutional layers (including pooling layers and activations), 1 flatten layer, 2 fully connected layers and 2 dropout layers. Here is a simple figure to show the structure of my model.

![ModelStructure][image6]

#### 2. Attempts to reduce overfitting in the model

I added drop out layers to the first and second fully connect layers, since these layers had lots of nodes and should be good to memory features even though some nodes were dropped out each time. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used flipping for augmentation, cropping for reducing computation, normalization for reducing error, and randomly dividing data into training set and validation set for reducing overfit.

For details about how I created the training data, see the next section. 

#### 5. Creation of the Training Set & Training Process

I used the recorded data for track 1, from all of the left, right and center cameras. Here are example images from center lane, left side and right side:

![center][image1] ![left][image2] ![right][image3]

And to augment my dataset, I flipped all these images, as well as steering angles. Here is an image that has then been flipped:

![left][image2] ![left_flip][image4]

For the images from the left, right cameras, I set a collection value to calculate corresponding steering angles based on steering angles in ¡®driving_log.csv¡¯, which were the steering angles for the center cameras. Here is the formula:

_angle_left = angle_center + correction_

_angle_right = angle_center ¨C correction_

The scale of the correction value actually affected the model¡¯s sensitivity to the departure angle. After several trials, I set the correction as 2, which gave the model appropriate sensitivity.Then, I attached corresponding steering angles as labels to all images.

After the collection process, I had 48216 of data points. I then preprocessed this data by moving out the top 70 rows and the bottom 25 rows of pixels in each image with Cropping2D layer and normalizing pixels¡¯ values with the formula (pixel/255.0)-0.5 to get 0 mean and small variance.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10, because the training results began to overfit dramatically at about 10th epochs, according to the graph of model mean squared error loss, which is showed below.

![MSE][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

