# response-ml

Machine Learning Responses:

## 7/7 Daily Response

1. Traditional programming takes an input of data and rules, and gives an output of answers. Machine learning, on the other hand, takes an input of answers and data, and gives an output of rules. In other words, from the given answers and data, it tries to link the two together.

2. The first answer I got was 12.999944, and the second was 12.999991. They are different because there is a limited amount of data to train on. However they are very very close to each other since they are a part of the training data, and therefore whatever pattern (line) is found, most likely fully applies to it as well. 

3. The best deal (not including things like remodeling) would be the Hudgins house since its price (per room) is about $137,000 less than predicted. The worst  deal would be the house of Church since it is about $100,000 more than predicted.

## 7/8 Daily Response

1. The training data is used to train the neural network, the testing data (remaining data) is used to see how well the network performs on other images/data that is new/not familiar. The purpose is to teach the computer to pick up patterns/equations, and be able to use them independently. The images variables are pixel arrays while the labels are numbers which represent the clothing.

2. Relu helps ensure that there are no negative numbers. So it sets any number that is less than 0, to 0. Softmax is usually used in the last layer, and sets the largest value to 1, and the rest to 0. This helps find the most likely answer since less data needs to be sorted through. There are 10 neurons in the last layer because there were 10 pieces of clothing, they are the output. Each neuron gives the probability that its correlated article is in the class.

3. The loss function finds the amount of error and works to minimize it. The optimizer then looks at the results from the loss function, and attempts to make another estimate with less error. The loss function will then analyze it, and the cycle continues. The loss function is categorical because the items which are being analyzed are also categorical. You cannot really measure them continuously.

4a. What is the shape of the images training set:​(60,000,28,28)

4b. What is the length of the labels training set?​6000

4c. What is the shape of the images test set? (1000,28,28)

4d. Estimate a probability model and apply it to the test set in order to
produce the array of probabilities that a randomly selected image is
each of the possible numeric outcomes:
array([1.5539440e-09, 8.2805976e-12, 4.7423885e-07, 4.3320365e-06,
1.9139490e-12, 2.6702085e-10, 1.1747801e-14, 9.9999344e-01, 1.0673884e-06, 7.2961376e-07], dtype=float32)

4e. Use np.argmax() with your predictions object to return the numeral with the highest probability from the test labels dataset: 7

4f. ![](sc7.8.png)


## 7/9 Daily Response

1. TF hub is a hub of datasets in the cloud that we are using and accessing. We bring it down to our local computers, and then use it’s data of movie reviews for our own modeling purposes.

2. The loss function we used is ‘BinarlyCrossentropy’. Loss functions measure the amount of error, and work to decrease it with each epoch, therefore bettering the model. The optimizer function used was ‘adam’. This works with the loss function by creating new estimations based on the output of the loss function. The loss function then analyses the new estimation and a cycle begins. The model turned out pretty well, but not great. It had an accuracy of 0.857 and a loss of 0.321.

3. The graph below demonstrates my training and validation loss. It decreases which shows that the model is improving, and there are fewer errors. The training loss performed better than the validation loss (you can tell because the dots are lower than the line as you move towards 10 epochs), but that is to be expected.
![](sc7.9loss.png)

4. The below graph is my training and validation accuracy graph. This is increasing, since the accuracy increases with each epoch. The validation accuracy did not perform as well as the training accuracy, but this is to be expected. You can tell because the validation accuracy line is not smooth and jets out in places. This is not to say it is not accurate -- it still generally follows the training accuracy.
![](sc7.9acc.png)

## Project 1

link to project 1 video: https://youtu.be/7sHHVERLgNw

1. In general, yes, it was effective at detecting potential violations. However, I would not recommend it be used for official purposes as actual distances were not accurately portrayed.

2. I think that this approach as a concept sounds more useful than it really is. As mentioned under the “Limitations and future improvements” section, the detector and camera calibration are not perfectly coordinated. This difference causes a gap between distances in real life, and the distance that is detected in the output video. It would also help if the camera were higher, this would make it easier to tell differences in space. Other factors, such as wind would also affect potential infections. Though most experts agree that Covid spreads through tiny droplets, they are still debating whether it is airborne This is something that can be recorded (direction and mph), but is very hard to incorporate into the computer's calculations since wind is so uncontrollable  and unpredictable. Who is to say what a certain gust would do? I think this approach would be helpful for cities. Many are attempting to reopen, and by using this camera and detector, officials could see general trends about whether people are following social distancing rules or not. This could help them predict the number of hospitalizations in the next few weeks, as well as decide if their citizens are responsible enough to handle a reopening. 

3. I would apply all ideas in the “Limitations and future improvements” section. For privacy reasons, I would probably try to blur out any detected faces. Based on the US’s response thus far, the government will not be tracking down specific people who were caught violating social distancing on such cameras. As mentioned in class, it is slightly disturbing to realize after the fact that you were being recorded -- this would help clear away any privacy concerns. 

## 7/14 Daily Response

A.
filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
weight  = 1
The first filter did not emphasize any specific kind of line as the 2nd and 3rd ones did. This image turned out much darker than the others, as if all features were not emphasized, but decreased. 

![](1.1.png)

filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
weight  = 1
This filter emphasizes horizontal lines in the image. Just by looking at it, you can tell as the wood supporting the railing is well defined -- much more than in the other images. In addition, in the line ‘filter’ you can see that one bracket contains only 0’s. This indicates that the filter will highlight horizontal features.

![](1.2.png)

filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
weight  = 1
This filter emphasizes vertical lines in the image. Just by looking at it, you can tell since the structure of the level above where the people are climbing is more defined. In addition, in the line ‘filter’ you can see that each bracket contains 0 as it’s middle number. This indicates that the filter will highlight vertical features. 

![](1.3.png)

When you apply a filter to the original array, you are multiplying it with a cluster of pixels (in this case 9 pixels). The multiplication values are then added to each other. Next, the same process happens to the next cluster of pixels. The filter you chose changes which features are highlighted or hidden. Computer vision is improved by convolutions since convolutions detect features and match them to labels. They highlight certain features of an image, which helps when trying to decipher one thing from another as well as analysis.


B. I applied a 2x2 filter to my convoluted image that shows clear vertical edges. When I applied this filter, it simplified the image by taking clusters of pixels as an input, and then outputting whichever is the largest within the cluster. This decreases the size of the image, but the output image keeps features that were before very clear, maintaining the image. In this example, the clusters are groups of 4. You can tell because in the code, when makinging the list ‘pixels’, x and y are used as a base, and at one point in the list each variable gest a ‘+1’ added to it twice, resulting in 4 pixels (from base, take a step vertically, horizontally, and diagonally to make a square). This method is useful as it simplifies images without letting go of the message of the image. If processing big data, this simplification could help decrease the amount of time a computer will be calculating and analyzing an image. 

![](pooling.png)
 
C. After training and comparing my DNN and CNN output, I was able to improve my model by adding the Conv2D and MaxPooling2D layers. After 10 epochs my DNN had a test accuracy of 0.9771. My CNN had an accuracy of 0.9982. By editing the convolutions by changing the neurons in each layer (32s to 16s and 64s) the more neurons, the longer the longer the training time. In addition, the more neurons, the better the accuracy, however only by a very small amount. If you add more convolutions, more weights are added. This can become problematic as the output (test data) could become more overfit and hence less accurate. The below graph helps visualize convolutions and demonstrates common features such as lines and curves.

![](c.png)

## 7/15 Daily Response
Horses and humans
1. You use the  ImageDataGenerator() command to pass directories through. The command will then auto label the images based on the directory name. For it’s arguments, you can use ‘rescale’ to normalize the image. You need to specify the directory, the size of the images (this is ‘target_size’) since you are not guaranteed to have multiple images of the same sizes, the batch size, and the class mode. In this example, we used binary, but if there were more than 2, you use categorical. The difference between doing this to the training vs validation datasets is simply the name. The process is the exact same. 
2. The number of layers is mostly personal preference, although it is encouraged to play around with it a bit in order to find the best results. In my example, as the size of the image decreases, the number of filters increases. The image sizes decrease thanks to the MaxPooling command. This takes a cluster of pixels, chooses the largest one, then uses that as the output. This image is then scanned and analyzed by the next line, which is usually a convolution, or if you choose to stop the cycle, a line which flattens the layer. The activation function which is the output is ‘sigmoid’. This pushes the value in one class towards 0, and the other (in another class) towards 1 (horse or human). This is used since the output also says that there is one output. This is okay since it is all binary and the ‘sigmoid’ activation is present. For the compiler, I used ‘binary_crossentropy’ for loss again. For the optimizer, I used ‘RMSprop()’. This is able to take parameters for the learning rate.

Regression
1. Because the auto.mpg pairwise plot contains 16 plots comparing different attributes, it is very useful for understanding the relationship of the attributes as well as the data. It is good for analysis of the interaction between variables because you can clearly view patterns and ranges in which the data resides. The diagonal access represents the comparison of values to themselves. The pairplot describes that as cylinders increase, the MPG slightly decreases. According to this plot, the peak number of cylinders in relation to MPG is 4 or 5. The relation between MPG and Displacement as well as MPG and Weight is the same as MPG and Cylinders, except much more continuous. The plots describing Cylinders and Displacement as well as Cylinders and Weight are very similar, with both increasing as the number of Cylinders increases. The last relationship, Weight and Displacement shows that as the Weight increases, so does displacement. Only plots with Cylinder as a value were not continuous. 
2. Interestingly enough, line 996 seemed to perform the best. It had the lowest values for almost every category (meaning the lowest amount of loss, absolute error, and mean error). The three lines after line 996 do not perform as well. Also, the training data definitely performed better than the validation data (you can tell since numbers in categories beginning with “val_” are usually larger than their corresponding columns). The plots do demonstrate this. It is much easier to see the difference between the training and validation data as the solid and dotted lines are easier to differentiate than simply numbers. In addition, the histogram which describes prediction errors shows that errors drop off completely, before happening again. The predicted values in the top right corner also vary more than in the middle of the graph. 

![](2.2.1.png)
![](2.2.2.png)

Overfit and underfit
1. When comparing the tiny, small, medium and large models, we can clearly see which performs best on testing data without overfitting or underfitting much. We can tell by the similarity of the solid and dotted lines. The closer and more similar they are, the better the model performed. In this case, it is tiny. It also helps demonstrate at which epochs things begin to go amiss, as most of the time validation lines curve upwards drastically. 

![](overfitandunderfit.png)











