"""
--------------------------------------------------------------------------
TensorFlow Basics
--------------------------------------------------------------------------
Go over very basic definitions and operations using the TensorFlow
Open-Source Library.
--------------------------------------------------------------------------
Gianluca Capraro
Created: May 2019
--------------------------------------------------------------------------
- Show how to create a simple constant using TensorFlow
- Create a TensorFlow Session, a class for running TensorFlow operations
- Run simple TensorFlow operations
- Learn to create placeholders for expected values
- Perform Matrix Multiplication with TensorFlow
--------------------------------------------------------------------------
"""

#import the tensorflow library
import tensorflow as tf

"""
--------------------------------------------------------------------------
Create Constants in TensorFlow
--------------------------------------------------------------------------
"""

#create instance of a tensorflow constant
hello = tf.constant('Hello World')

#print the type to verify
print("\nTensorFlow Constant 'Hello' Type:")
print(type(hello))
print('\n')

#another simple constant example
x = tf.constant(100)

#print this object details
print('TensorFlow Constant X:')
print(x)
print('\n')

"""
--------------------------------------------------------------------------
Create TensorFlow Session
--------------------------------------------------------------------------
"""

#create tensorflow session
print('Created TensorFlow Session.')
tf_session = tf.Session()

#run the constants in the session
print("\nRunning TF Session for Object 'Hello':")
print(tf_session.run(hello))
print('\n')
print('Running TF Session for X:')
print(tf_session.run(x))
print('\n')


"""
--------------------------------------------------------------------------
Running Simple TensorFlow Operations
--------------------------------------------------------------------------
"""

#create some additional constants to perform tf operations
x = tf.constant(2)
y = tf.constant(3)

#perform operations in tf session
with tf.Session() as tf_session:
	print('Operations with TensorFlow Constants:')
	print('Addition: ', tf_session.run(x+y))
	print('Subtractino: ', tf_session.run(x-y))
	print('Multiplication: ', tf_session.run(x*y))
	print('Division: ', tf_session.run(x/y))
	print('\n')



"""
--------------------------------------------------------------------------
Creating Placeholders
--------------------------------------------------------------------------
"""

#create two placeholders that expect an object of type int32
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)


#define operations that can have variable inputs
#analagous to functions that expect a certain input
add = tf.add(x,y)
sub = tf.subtract(x,y)
multi = tf.multiply(x,y)

#create the dictionary of x,y vals that will be used
d = {x:20,y:30}

#perform operations with placeholders
#use feed_dict to input x and y values
with tf.Session() as tf_session:
	print('Operations with Placeholders:')
	print('Addition: ', tf_session.run(add,feed_dict=d))
	print('Subtraction: ', tf_session.run(sub,feed_dict=d))
	print('Multiplication: ', tf_session.run(multi,feed_dict=d))

	print('\n')


"""
--------------------------------------------------------------------------
Matrix Multiplication with Numpy and TensorFlow
--------------------------------------------------------------------------
"""

#import numpy to create matrices and arrays
import numpy as np

#create a 2x1 array, and a 1x2 array 
a = np.array([[5.0,5.0]])
b = np.array([[2.0],[2.0]])

#create 2 TF matrices that can be used with TensorFlow
matrix1 = tf.constant(a)
matrix2 = tf.constant(b)

#define matrix multiplication operation on the 2 TF matrices
matrix_multi = tf.matmul(matrix1,matrix2)

#run the session to perform the operation
with tf.Session() as tf_session:
	print('Matrix Multiplication TensorFlow Operation Result:')
	mat_result = tf_session.run(matrix_multi)
	print(mat_result)
	print('\n')
