"""
--------------------------------------------------------------------------------------------------
Using the MNIST dataset along with a Multi-Layer Perceptron Model to classify Hand-Written Numbers
--------------------------------------------------------------------------------------------------
The MNIST Dataset is a collection of arrays that represent hand-
written digits 0-9 using pixels. In this project, TensorFlow will be
used to classify the written number by training on the array
values in MNIST.
--------------------------------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
"""

#import libraries
import tensorflow as tf

#import the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

#read in the mnist dataset from current folder
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Quick Aside on One-Hot encoding
-------------------------------
With one hot set to True, this means that the actual target
(label) is one hot encoded. This can be seen in the quick 
example below:

print(mnist.train.labels[0])

This will print an array of length 10, with all zeros and a 1
in one column. The 1 value indicates the true label of the handwritten
number. For example, if the array has a 1 at index 1, the number
is a 1. 
"""

#confirm data read by viewing the type of mnist
print('\nMNIST Read Successfully.\nMNIST Read Data Type:')
print(type(mnist))
print('\n')

#check the number of training and testing examples in the dataset
print('Number of Training Examples:')
print(mnist.train.num_examples)
print('\n')
print('Number of Testing Examples:')
print(mnist.test.num_examples)
print('\n')

"""
--------------------------------------------------------------------------------------------------
Visualize the MNIST Dataset
--------------------------------------------------------------------------------------------------
The dataset contains 'images' represented as 784x1 arrays that each
correspond to a different number (0-9) that is handwritten. 

Prior to setting up our prediction algorithm, let's see what an example image
looks like to us, and and example of what we want to pass through the neural network.
--------------------------------------------------------------------------
"""

#import matplot lib for plotting
import matplotlib.pyplot as plt

#reshape one of the images in the training set so it is in its original 28x28 form
example_image = mnist.train.images[1].reshape(28,28)

#plot this image using matplotlib to see it, it clearly shows a 3
#viewing this image we can get an idea of all the information within this dataset and what it represents
print('Showing Example of Single Image Within MNIST Dataset...')
plt.imshow(example_image, cmap='gist_gray')
plt.title('Example Image of Hand-Written Number in Dataset')
plt.show()
print('\n')

#print the example_image array
#we can see that non-zero values correspond to brighter pixel values
print('Representation of Example Image as Original Array:')
print(example_image)
print('\n')


"""
--------------------------------------------------------------------------------------------------
Passing Data into the Neural Network
--------------------------------------------------------------------------------------------------
Although it is easy for us to recognize the 3 in the plot, this task would
be much more difficult for the computer. Instead of the 28x28 matrix representation
of the image, we will pass in the 784x1 array which will be easier for the 
computer to process and understand.
--------------------------------------------------------------------------------------------------
"""

#reshape the matrix to 784x1
example_image = mnist.train.images[1].reshape(784,1)

#Show the 784x1 array that will be used
print('Showing Example of 784x1 Image Array to be Processed, MNIST Dataset...')
plt.imshow(example_image, cmap='gist_gray', aspect = 0.03)
plt.title('Visual Representation of Array to be Processed')
plt.show()
print('\n')


"""
--------------------------------------------------------------------------------------------------
Creating the Model to Classify MNIST Numbers based on Array Values
--------------------------------------------------------------------------------------------------
Although it is easy for us to recognize the 3 in the plot, this task would
be much more difficult for the computer. Instead of the 28x28 matrix representation
of the image, we will pass in the 784x1 array which will be easier for the 
computer to process and understand.
--------------------------------------------------------------------------------------------------
"""

#create x as a placeholder, pass in float as the data type, [none,784] as shape
#the value for shape is chosen because we have yet to decide our batch size, but we know that each array will be 784
x = tf.placeholder(tf.float32,shape=[None,784])

#define the weights that will be used for training
#the shape will be 784 (length of each array) by 10 (the number of possible hand-written numbers or our target labels)
W = tf.Variable(tf.zeros([784,10]))

#define the biases of our model, should match up w second variable in W zeros
b = tf.Variable(tf.zeros([10]))

#create the formula for y
#this is our formula to predict the target or label value
y = tf.matmul(x,W) + b

#define where we will pass in labels that are correctly predicted
y_true = tf.placeholder(tf.float32,shape=[None,10])

#reduce error between y_true and our predicted y using cross entropy
#how we are actually defining the error
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true, logits = y))

#optimize using Gradient Descent
#this is the method of how we will try to reduce error
#lower learning rate usually increases accuracy but may take longer to train
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)

#using our optimizer, train and minimize the error we have defined
train = optimizer.minimize(cross_entropy)

#initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as tf_session:
	
	#initialize all the variables
	tf_session.run(init)

	#how many times should the various batches be fed into the model
	for step in range(1000):

		#grab batch of 100 numbers, the array, and the target values
		batch_x, batch_y = mnist.train.next_batch(100)

		#run the tf session and pass the x and y values into the placeholders
		tf_session.run(train, feed_dict={x : batch_x, y_true : batch_y})

	#where is y equal to y_true
	matches = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))

	#calculate the accuract of matches
	accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

	# determine accuracy based on our test set
	print('\nAccuracy of Hand-Written Number Determination:')
	print(tf_session.run(accuracy, feed_dict={x : mnist.test.images, y_true : mnist.test.labels}))
	print('\n')


"""
Running this model, we can see around a 92% accuracy in determining the
hand-written number from its numerical array representation.

Going forward, we will do all of the above using the TensorFlow Estimator 
to quickly run and create models similar to this one, in a simpler fashion.
"""



