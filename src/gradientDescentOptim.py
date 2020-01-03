import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# generate a random number for Weights
# these two commands are similar function
#Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Weights = tf.Variable(np.random.rand(10).astype(np.float32)[5])

# set biases values are all zeros
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# cost function
loss = tf.reduce_mean(tf.square(y-y_data))

# set optimizer using Gradient Descent method
# learning rate is 0.5 here, should be less than 1
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# initialize
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)          # Very important

for step in range(201):
# run by each step
    sess.run(train)
# outputs by every 20 steps
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
print('computation is end')
