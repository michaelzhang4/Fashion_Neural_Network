# Fashion Neural Network #

In this project I created a Neural Network to identify pieces of fashion given an image.
I used the keras fashion dataset which is loaded in as training and test data. I then created a model with 3 layers, one layer flattening the data then two dense layers which looks at the pixels in an image and returns percentage of likelihood of each possible item.
The model is trained with the training data and then when the project is run you can see how accurate the model is by validating what the model predicts the test data to be in comparison to the actual value. 
