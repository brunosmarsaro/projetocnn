"""
The idea here is to implement a neural network with similar structure to, the already known, AlexNet from the ImageNet
Competition 2012.
The NN has 8 main layers, being the first 5 convolutional and the 3 left fully connected layers.


    author: Bruno Smarsaro Bazelato
"""
import tensorflow as tf
from nn import DeepNN

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
#config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

nw = DeepNN()
if nw.inputs == -1: exit()

max_steps = 10000
batch_size = 20

# First Convolutional Layer

W_conv1 = nw.weight_variable([5, 5, 1, 32]) # 55, 55, 5, 96
b_conv1 = nw.bias_variable([32])

x_image = tf.reshape(nw.x, [-1,32,32,1]) # [-1,1024,1024,5]

h_conv1 = tf.nn.relu(nw.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = nw.max_pool_2x2(h_conv1)

# Second Convolutional Layer

#W_conv2 = nw.weight_variable([27, 27, 96, 256])
#b_conv2 = nw.bias_variable([256])

#h_conv2 = tf.nn.relu(nw.conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = nw.max_pool_2x2(h_conv2)

# Third Convolutional Layer

# W_conv3 = nw.weight_variable([13, 13, 256, 384])
# b_conv3 = nw.bias_variable([384])
#
# h_conv3 = tf.nn.relu(nw.conv2d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = nw.max_pool_2x2(h_conv3)
#
# # Fourth Convolutional Layer
#
# W_conv4 = nw.weight_variable([13, 13, 384, 384])
# b_conv4 = nw.bias_variable([384])
#
# h_conv4 = tf.nn.relu(nw.conv2d(h_pool3, W_conv4) + b_conv4)
# h_pool4 = nw.max_pool_2x2(h_conv4)
#
# # Fifth Convolutional Layer
#
# W_conv5 = nw.weight_variable([13, 13, 384, 256])
# b_conv5 = nw.bias_variable([256])
#
# h_conv5 = tf.nn.relu(nw.conv2d(h_pool4, W_conv5) + b_conv5)
# h_pool5 = nw.max_pool_2x2(h_conv4)

# First Fully Connected Layer

W_fc1 = nw.weight_variable([3136, 1024])
b_fc1 = nw.bias_variable([1024])

h_pool1_flat = tf.reshape(h_pool1, [-1, 3136]) #7*7*256
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# Second Fully Connected Layer

# W_fc2 = nw.weight_variable([8192, 4096])
# b_fc2 = nw.bias_variable([4096])
#
# h_pool2_flat = tf.reshape(h_pool1_flat, [-1, 8192]) # 7*7*64
# h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)

# Third Fully Connected Layer (Readout Layer)

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc3 = nw.weight_variable([1024, 2])
b_fc3 = nw.bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# Train and Evaluate the Model

cross_entropy = tf.reduce_mean(-tf.reduce_sum(nw.y_ * tf.log(y_conv), reduction_indices=[1]))
print(tf.rank(cross_entropy))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#print(tf.rank(tf.argmax(y_conv,1)))
#print(tf.rank(tf.argmax(nw.y_,1)))
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(nw.y_,1))
#print(correct_prediction)
#input()
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

print("Train and Evaluate the Model")

for i in range(max_steps):
    batch = nw.next_batch(nw.inputs["training"], batch_size)
    train_step.run(feed_dict={nw.x: batch[0], nw.y_: batch[1], keep_prob: 0.5})
    if i%100 == 0:
        #print(batch[0].shape, type(batch[0]))
        #print(batch[1].shape, type(batch[1]))
        #input()
        train_accuracy = accuracy.eval(feed_dict={nw.x: batch[0], nw.y_: batch[1], keep_prob: 1.0})

        print("step %d, training accuracy %g"%(i, train_accuracy))


print("test accuracy %g"%accuracy.eval(feed_dict={
    nw.x: nw.inputs["testing"][0], nw.y_: nw.inputs["testing"][1], keep_prob: 1.0}))

sess.close()