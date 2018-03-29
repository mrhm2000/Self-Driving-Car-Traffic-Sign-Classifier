# Build a Traffic Sign Recognition Project 


The goals of this project is to design a model can be use to recognize traffic sign. German traffic sign are using on this exercise. Steps taken as follow :
* Load, summarize and visualize data set
* Data set preprocessing 
* Design, train and test a model architecture
* Optimize architecture and parameter
* Analyze new traffic sign and went trough same data preprocessing
* Visualize the Softmax probabilities of the new images
* Summarize the results and report



## Load, summarize and visualize data set

I used pickle, cvs, numpy and matplotlib on this process.

signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 


<figure>

 <img align=“left” src="/private/data1.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


It is a bar chart showing how the data is distributed across the different labels.

<figure>
<center>
 <img align=“center” src="/private/data2.png" width=“700” alt="Combined Image" />
</center>
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


Data set preprocess is convert image to grayscale, normalize and generate random image from original data set to offset under presented classes show on above chart. 

<figure>
 <img align=“center”  src="/private/data3.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>



<figure>
 <img align=“center” src="/private/data4.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


Below is visualization of original, greyscale, normalized and rotated image. Images are normalize to (32,32,1) mean = 0.31.

<figure>
 <img align=“center”  src="/private/data6.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


---


## Design and Test a Model Architecture

#### Question 1:
*Describe how you preprocessed the data. Why did you choose that technique?*
##### Answer

Image preprocessing method I used consist of :
* Converted images to greyscale - original images (32,32,3) to grayscale (32,32,1). As suggested [here](https://stackoverflow.com/questions/20473352/is-conversion-to-gray-scale-a-necessary-step-in-image-preprocessing). processing image on grayscale tend to yield a faster processing compare to color. Another reference [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) from Pierre Sermanet and Yann LeCun.
* Normalized images as suggested on Udacity lesson 6.23 Introduction to tensorflow - Normalized input to [0:1]. We want input images to have 0 means to allow optimizer to get best result. Normalized image process resulting image mean around 0.31. Another reading reference [here](https://www.imagemagick.org/discourse-server/viewtopic.php?t=26148).
* Generated extended train images. Fig 1 and fig 2 show original images distribution. As you you can see on the chart, some classes under presented. This may lead to accuracy problem when model doesn't have adequate training sample. I decided generate some more images by rotation augmentation technique to achieve a well distribute images - Fig 4. Function def : selected_rotation used randomly selected angle from -20 to 20 and skip -1 to 1 degree.


#### Question 2:
2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)
##### Answer:

### Architecture


<table>
<tr><td>Layer</td>
<td>Name</td>
<td>Description</td></tr>
    
<tr><td>Layer 1: Convolutional</td>
    <td>conv1</td>
<td>1x1 stride, Valid padding, Input 32x32x1 ==> Output 28x28x6</td></tr>

<tr><td>Activation: ReLU </td>
        <td>conv1_relu</td>
<td>ReLU - a rectified linear unit</td></tr>

<tr><td>Pooling</td>
        <td>conv1_pool</td>
<td>2x2 stride, Valid padding, Input 28x28x6 ==> Output 14x14x6</td></tr>

<tr><td>Layer 2: Convolutional</td>
        <td>conv2</td>
<td>1x1 stride, Valid padding, Input 14x14x6 ==> Output 10x10x16</td></tr>

<tr><td>Activation: ReLU</td>
        <td>conv2_relu</td>
<td>ReLU - a rectified linear unit</td></tr>

<tr><td>Pooling</td>
        <td>conv2_pool</td>
<td>2x2 stride, Valid padding, Input 10x10x16 ==> Output 5x5x16</td></tr>

<tr><td>Flatten</td>
        <td></td>
<td>Fully connected layers. Input 5x5x16. ==> Output 400.</td></tr>

<tr><td>Layer 3: Fully connected</td>
        <td></td>
<td>Fully Connected. Input = 400. ==> Output = 200.</td></tr>

<tr><td>Activation: ReLU</td>
        <td></td>
<td>ReLU - a rectified linear unit</td></tr>

<tr><td>Dropout operation</td>
        <td></td>
<td>Dropout</td></tr>

<tr><td>Layer 4: Fully connected</td>
        <td></td>
<td>Fully Connected. Input 200. ==> Output 100.</td></tr>

<tr><td>Activation: ReLU</td>
        <td></td>
<td>ReLU - a rectified linear unit</td></tr>

<tr><td>Dropout operation</td>
        <td></td>
<td>Dropout</td></tr>

<tr><td>Layer 5: Fully Connected</td>
        <td></td>
<td>Fully Connected. Input 100. ==> Output = 43.</td></tr>


</table> 

#### Question 3:

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

##### Answer


I trained the model using Adam optimizer and Lenet architecture since augmented data is already normalized on grayscale. Following is a table show parameter used and accuracy. Note : Please refer to def Lenet section to find out original and experimental Lenet (mod Lenet). Experimental Lenet focus on different shape after second layer flatten process.


<table>
  <tr>
    <th>Architecture</th>
    <th>Epoch</th>
    <th>Batch Size</th>
    <th>Rate</th>
    <th>Mu</th>
    <th>Sigma</th>
    <th>Dropout</th>
    <th>Preprocess</th>
    <th>Validation Accuracy</th>
    <th>Test Accuracy</th>
    <th>Observation and Action</th>
  </tr>
   <tr>
    <td>Original Lenet</td>
    <td>10</td>
    <td>128</td>
    <td>0.0009</td>
    <td>0</td>
    <td>0.1</td>
    <td>1.0</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image </td>
    <td>97.40%</td>
        <td>86.20%</td>
    <td></td>
  </tr>
     <tr>
    <td>Mod Lenet</td>
    <td>10</td>
    <td>128</td>
    <td>0.0009</td>
    <td>0</td>
    <td>0.1</td>
    <td>1.0</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image </td>
    <td>97.50%</td>
        <td>85.90%</td>
    <td>It wasn't much different with orginal Lenet. Validation accuracy increased but test accuracy slighly decreased. I settled with this architecture for now and change more parameter.</td>
  </tr>
    
  <tr>
    <td>Mod Lenet</td>
    <td>25</td>
    <td>128</td>
    <td>0.0009</td>
    <td>0</td>
    <td>0.1</td>
    <td>0.5</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image </td>
    <td>99.30%</td>
        <td>93.40%</td>
    <td>Reach 93.4% test accuracy. </td>
  </tr>
   
    
   <tr>
    <td>Mod Lenet</td>
    <td>50</td>
    <td>128</td>
    <td>0.0009</td>
    <td>0</td>
    <td>0.1</td>
    <td>0.5</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image </td>
    <td>99.50%</td>
        <td>93.90%</td>
    <td></td>
  </tr>
    
   <tr>
    <td>Mod Lenet</td>
    <td>50</td>
    <td>128</td>
    <td>0.0008</td>
    <td>0</td>
    <td>0.1</td>
    <td>0.5</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image </td>
    <td>99.50%</td>
       <td>93.50%</td>
    <td></td>
  </tr>
  
  <tr>
    <td>Mod Lenet</td>
    <td>100</td>
    <td>128</td>
    <td>0.0008</td>
    <td>0</td>
    <td>0.1</td>
    <td>0.5</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image</td>
    <td>99.70%</td>
       <td>94.50%</td>
    <td></td>
  </tr>
      <tr>
    <td>Mod Lenet</td>
    <td>150</td>
    <td>128</td>
    <td>0.0008</td>
    <td>0</td>
    <td>0.1</td>
    <td>0.5</td>
    <td>Normalized, grayscale, randomize, rotate : generate additional image</td>
    <td>99.7%</td>
    <td>94.50%</td>
    <td></td>
  </tr>
</table>


#### Question 4:
* Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include

I started with LeNet architecture because of simplicity. After first iteration I observed the model tended to overshoot. I played with some parameter to get maximum validation accuracy (see observation table above). At the end I settle with Epoch 150, batch size, 128, learing rate 0.0008 and dropout rate=0.5. Final validation accuracy 99.7% and testing accuracy 95.00%.





### Test a Model on New Images

#### Question 1:

*Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*



<figure>
 <img align=“center”  src="/private/data5.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

Found some German traffic sign images online. The size, quality of images are vary with much more background. I run same pre-processing process on all images. Post process images shape is min : 0.08  max : 0.88  mean : 0.466399247199  variance : 0.0549784484242 shape: (5, 32, 32, 1) 
* 1st and 4th image has a black background.
* 2nd, 3rd, 5th and 7th has color background.
* 6th has a white background.
* 5th also slanted.

Model may have problem identify some image with more color background, shadows obstructing objects or slanted images. I pick images with those condition to see how model recognize them. 


#### Question 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Model able to recognized sign accurately even with white, black and color background. Model also able to identify slanted image accurately.

<figure>
 <img align=“center”  src="/private/data7.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


### Extra Images Model Accuracy Vs Prediction.


<table>
  <tr>
    <th>Images </th>
    <th>Correct Class ID</th>
    <th>1st Guess</th>
    <th>2nd Guess</th>
    <th>3rd Guess</th>
    <th> Note </th>
  </tr>
  <tr>
    <td>001.jpg</td>
    <td>Ahead Only (35) </td>
    <td>Ahead Only (35) 100%</td>
    <td></td>
    <td></td> 
    <td>1st guess is correct</td>
  </tr>
  <tr>
    <td>002.jpg</td>
    <td>Roundabout mandatory (40)</td>
    <td>Roundabout mandatory (40) 100%</td>
    <td></td>
    <td></td>
    <td>1st guess is correct</td>
  </tr>
  <tr>
    <td>003.jpg</td>
    <td>Right-of-way at the next intersection (11)</td>
    <td>Right-of-way at the next intersection (11) 100%</td>
    <td></td>
    <td></td>
    <td>1st guess is correct</td>
  </tr>
  <tr>
    <td>004.jpg</td>
    <td>Bumpy Road (22)</td>
    <td>Bumpy Road (22) 100%</td>
    <td></td>
    <td></td>
    <td>1st guess is correct</td>
  </tr>
  <tr>
    <td>005.jpg</td>
    <td>No Entry (17)</td>
    <td>No Entry (17) 100%</td>
    <td></td>
    <td></td>
    <td>1st guess is correct</td>
  </tr>
  <tr>
    <td>006.jpg</td>
    <td>Yield (13)</td>
    <td>Yield (13) 100%</td>
    <td></td>
    <td></td>
    <td>1st guess is correct</td>
  </tr>
  <tr>
    <td>007.jpg</td>
    <td>No Passing (9)</td>
    <td>No Passing (9) 100%</td>
    <td></td>
    <td></td>
    <td>1st guess is correct</td>
  </tr>
</table>

### Extra Images Softmax Probability

Below is Sofmax probability for German traffic sign 002.jpg, 003.jpg and 004.jpg. Model able to recognize class ID for each accurately.

<figure>
 <img align=“center”  src="/private/data8.png" width=“700” alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

