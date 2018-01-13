# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[img_vis]: ./output/visualization.png "Visualization"
[img_his]: ./output/histogram.png "Visualization"
[img_aug]: ./output/augment.png "Before/after augmentation"
[image4]: ./data/additional/1.png "Traffic Sign 1"
[image5]: ./data/additional/11.png "Traffic Sign 2"
[image6]: ./data/additional/12.png "Traffic Sign 3"
[image7]: ./data/additional/18.png "Traffic Sign 4"
[image8]: ./data/additional/23.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

##### Sample traffic sings and their names:
![alt text][img_vis]
##### Histogram of the signs in the training data:
![alt text][img_his]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Each image is augmented via random rotation and cropping. The rotation angle is sampled from -15 to 15 degrees and the cropping displacement is sampled from -2 to +2 pixels in either direction.

#####  A traffic sign image before and after augmentation.
![alt text][img_aug]

For each training image, four augmented images are generated. As such, the total number of images in the training increased by a factor of five. 

In addition, image-wise standardization is performed during training and test. Each input image is subtracted by the image-wise mean and divided by the image-wise standard deviation.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consists of the following layers:

| Layer         		    |     Description	        					            | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x3 RGB image   							            | 
| Input + Grayscale     | 32x32x4 RGB-Gray image   							       | 
| Channel-wise Dropout	|	32x32x4 keep probability = 3/4 layers         |
| Convolution 3x3       | 32x32x8          	                            |
| ReLU					        |	32x32x8         			                        |
| Max pooling	          | 16x16x8                               				|
| Residual block        |	16x16x8         			                        |
| Residual block        |	16x16x8         			                        |
| Residual block        |	16x16x8         			                        |
| Residual block        |	16x16x8         			                        |
| Contract block        |	8x8x32          			                        |
| Residual block        |	8x8x32          			                        |
| Residual block        |	8x8x32          			                        |
| Residual block        |	8x8x32          			                        |
| Residual block        |	8x8x32          			                        |
| Contract block        |	4x4x128          			                        |
| Bottleneck block      |	4x4x128          			                        |
| Bottleneck block      |	4x4x128          			                        |
| Bottleneck block      |	4x4x128          			                        |
| Bottleneck block      |	4x4x128          			                        |
| Contract block        |	2x2x512          			                        |
| Global averaged pool  |	1x1x512          			                        |
| Dropout	              |	1x1x512 keep probability = 0.5 * (1 + p)      |
| Fully connected       |	1x1x256          			                        |
| Dropout	              |	1x1x256 keep probability = p                  |
| Fully connected		    |	1x1x43          			                        |
| Softmax				        |	1x1x43          			                        |

The Residual and Bottleneck block follows the architecture of [He et al.](https://arxiv.org/abs/1512.03385). The contract block follows the idea of [Huang et al.](https://arxiv.org/abs/1608.06993), consisting of 3x3 convolution filters with stride of 2x2 and then concatenated with a max pooling of the input layer. Batch normalization is not used in this work.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with exponentially decaying learning rate. The learning rate starts with 2e-3 and is reduced by 10% every 680 steps. The batch size is chosen to be 256 images and 20 epochs are used to train the model. The p value determining the keep probability of the dropout layer is chosen to be 0.25.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.0%
* test set accuracy (top 1) of 96.3% 
* test set accuracy (top 5) of 98.9% 

The architecture follows the idea of ResNet, which has seen great success in image classification benchmarks. The training set accuracy is found to be saturated with networks of fewer layers. However, increasing the network capacity improves the generalization of the model. The drop out layer is also found to help reduce the validation error. In the work of [Sermanet et al.](http://ieeexplore.ieee.org.stanford.idm.oclc.org/abstract/document/6460867/), the usage of grayscale images instead of color images results in better results. Following this idea, the grayscale channel is added to the RGB data. However, no significant improvement has been observed. A channel-wise dropout layer is then added, forcing the network to learn the complementary nature between these color channels, which successfully improves the validation accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The background of the 1st, 4th, and 5th images are quite busy and the priority road sign not forward facing.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		      | 30 km/h   									| 
| Right-of-way     			| Right-of-way 										|
| Priority road					| Priority road											|
| General caution	      | General caution					 				|
| Slippery Road			    | Slippery Road      							|


The model was able to correctly predict 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Image                	|     Probability of correct sign	        			|
|:---------------------:|:---------------------------------------------:|
| 30 km/h      		      | 100					                              		|
| Right-of-way     			| 100					                        			  	|
| Priority road					| 100					                        					|
| General caution	      | 100					 				                          |
| Slippery Road			    | 100    						                       	    |

The probability of the rest of the top five predictions are all below 1e-10.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
