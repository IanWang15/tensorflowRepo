import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# used to add layer including hidden layer and output layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
# y = Weights * x + biases
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
# apply activation function to y
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# initial data is better not zeros
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# it is not necessary to have placeholder.
# here is just for easy to change
# None is able to accept different shapes of data
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# first layer: input is 1 neuron; output is 10 neurons.
# activation function is relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# output layer: input is 10 neuron from l1; output is 1 neurons.
# activation function is None
prediction = add_layer(l1, 10, 1, activation_function=None)

# cost function
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# optimizer is Gradient Descent
# learning rate is 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialize
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

