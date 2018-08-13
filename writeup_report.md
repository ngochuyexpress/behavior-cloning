# **Behavioral Cloning**

### Huy Nguyen
---

**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[corner]: ./corner.png "Corner"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submitted files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run1.mp4 recorded video of the model driving 1 lap

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed


My model consists of a convolution neural network with 4 convolutions layers with filters of sizes (5,5), (4,4), (5,5) and (3,3) respectively and depths between 24 and 64 (model.py lines 64-70). After that, it follows by 4 dense layers of output sizes 100, 50, 10 and 1 respectively. Between the convolutional layers and dense layers is a Flatten layer.

The model includes RELU layers to introduce nonlinearity and max pooling layers to reduce dimentions after each convolutional layer.

The data is normalized in the model using a Keras lambda layer (model.py line 62), and cropped (model.py line 63).

#### 2. Attempts to reduce overfitting in the model

No special treatment is needed to reduce overfitting in the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

I use the provided training data. During training, I noticed that my models keeps failing to make left turn in a corner:
![alt_text][corner]

A quick glance at the training data, I found that there is not enough data of this corner provided. So I generated more data of driving through this corner and add them into the training set.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia model for self driving car. I thought this model might be appropriate because it gives a good initial result without much modification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a comparable loss between training set and validation set so overfitting is not a big concern here.

The main issue that I have is that the model does not do well in the simulator. It seems that the issue is in the training data.

Then I take some simple steps to improve training data like normalization (model.py line 62), cropping (model.py line 63), and flipping the image randomly to avoid biased to left turns (model.py line 42-44). It improved the driving quality but still far from acceptable. The vehicle still got too close to the side of track or fell off the track completely at some points.

I took closer look at the data and found that in majority training sample, the angle is 0. This may causes the model to bias toward not steering at all (which I also noticed in the simulator). To avoid this bias, I augment the data by randomly pick side camera rather than default to the center camera. In addition, when the angle is too small, I  added a logic to force using side camera only (model.py line 30-32). By doing this, the model is more proactive in trying to get the car back to the center of the track than letting driving straight to the side.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. For example, this corner: ![alt_text][corner]

To improve the driving behavior in these cases, I generated more training data manually in these spots and add them into the training set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-80) consisted of a convolution neural network with the following layers and layer sizes :

4 convolution layers with filter sizes of (5,5), (4,4), (5,5) and (3,3) respectively and depths are between 24 and 64.
Each convolution layer is followed by a RELU activation layer and max pooling layer.
A flatten layer.
4 dense layers of output size of 100, 50, 10 and 1 respectively.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Failed approaches
1. I tried converting the image to grayscaling. However, it does not seem to help much. My guess is that at some spots in the track, the side  and lane colors are too similar  in grayscale making it even harder to disguish the lane.
2. I tried adding dropout layers after each dense layers. This does not seem to help much.

#### 4. Creation of the Training Set & Training Process

I used  the provided training data with some additional data generated manually.
