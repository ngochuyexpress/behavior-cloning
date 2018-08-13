# **Behavioral Cloning**

### Huy Nguyen
---

**Behavioral Cloning Project**

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[corner]: ./corner.png "Corner"

---
### Files Submitted & Code Quality

#### 1. Submitted files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 recorded video of the model driving 1 lap

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia model for self driving car. I thought this model might be appropriate because it gives a good initial result without any modification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a comparable loss between training set and validation set so overfitting is not a big concern here.

First, I took some simple steps to improve training data like normalization (model.py line 62), cropping (model.py line 63), and flipping the image randomly to avoid biased to left turns (model.py line 42-44). It improved the driving quality but still far from acceptable. The vehicle still got too close to the side of track or fell off the track completely at some points.

The main issue that I have is that while the loss seems low for both trainning set and validation set, the model does not do well in the simulator. It seems that the issue is in the training data.

I took closer look at the data and found that in majority training samples, the angle is 0. This causes the model to bias toward 0-angle, i.e., not steering at all. I also noticed this behavior while watching the simulator. To avoid this bias, I augment the data by randomly picking side camera rather than default to the center camera. In addition, when the center angle is too small (close to 0), only side camera is used (model.py line 30-32). By doing this, the model is more proactive to steer and always try to get the car back to the center of the lane.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. For example, this corner: ![alt_text][corner]

To improve the driving behavior in these cases, I generated more training data manually in these spots and add them into the training set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-80) consisted of a convolution neural network with the following layers and layer sizes :

1. The data is normalized in the model using a Keras lambda layer (model.py line 62), and cropped (model.py line 63).

2. 4 convolution layers with filter sizes of (5,5), (4,4), (5,5) and (3,3) respectively and depths are between 24 and 64.
Each convolution layer is followed by a RELU activation layer to introduce nonlinearity and max pooling layer to reduce dimentions.

3. A flatten layer.

4. 4 dense layers of output size of 100, 50, 10 and 1 respectively.

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Failed approaches
During training and searching for the right model, I tried different approaches and failed. Although these approaches are not included in my final solution. Trying them and seeing how they worked help me learn more about these techniques and their limitations.

1. I tried converting the image to grayscaling. However, it does not seem to help much. My guess is that at some spots in the track, the side  and lane colors are too similar  in grayscale making it even harder to disguish the lane.
2. I tried adding dropout layers after each dense layers. This does not seem to help much.

#### 4. Creation of the Training Set & Training Process

I used  the provided training data with some additional data generated manually.
