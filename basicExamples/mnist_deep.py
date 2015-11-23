import input_data
import tensorflow as tf

# http://tensorflow.org/tutorials/mnist/pros/index.html

L1_SIZE = 32
L2_SIZE = 64
L3_SIZE = 1024

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Layer 1
# 32 features for each 5x5 chunk
W_conv1 = weight_variable([5, 5, 1, L1_SIZE])
b_conv1 = bias_variable([L1_SIZE])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Layer 2
# 64 features for each 5x5 chunk
# Why so many weight variables?  Shouldn't it just be one 32 -> 64 mapping for each 5x5 chunk?
W_conv2 = weight_variable([5, 5, L1_SIZE, L2_SIZE])
b_conv2 = bias_variable([L2_SIZE])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Layer 3
# 1024 Neurons that each get input from each of the 64 features from each of the 7 * 7
# subimages
W_fc1 = weight_variable([7 * 7 * L2_SIZE, L3_SIZE])
b_fc1 = bias_variable([L3_SIZE])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*L2_SIZE])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout - technique to avoid overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax aggregation / readout layer
W_fc2 = weight_variable([L3_SIZE, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and evaluate
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_size = len(mnist.test.images)
lower = 0
upper = 256
accuracy_ = []
sizes = []

while upper < test_size + 256:
  u = min(upper, test_size)
  accuracy_.append(accuracy.eval(feed_dict={
    x: mnist.test.images[lower: u], y_: mnist.test.labels[lower: u], keep_prob: 1.0}))
  sizes.append(u-lower)
  lower += 256
  upper += 256

total = 0
for a, s in zip(accuracy_, sizes):
  total += a * s

print "test accuracy %g"%(total / float(len(mnist.test.images)))
