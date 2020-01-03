import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)
print(type(product))
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

# not able to generate session using numpy in this case
#error message: Can not convert a ndarray into a Tensor or Operation.
m1 = [3,3]
m2 = [[2], [2]]
prodt = np.dot(m1, m2)

with tf.Session() as sess:
    results = sess.run(prodt)
    print(results)

print('computation is end')
