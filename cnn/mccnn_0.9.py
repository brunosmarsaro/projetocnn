import timeit

start = timeit.default_timer()

# Variáveis e Parametros

database_path = "database_28x28"
max_steps = 500
batch_size = 10
index = 0
testing_percentage = 10
validation_percentage = 0
width = 28
height = 28
n_classes = 2
n_maxpool = 1

# Carregar base de dados
import input

inputs = input.Process(path=database_path)
images = inputs.run(testing_percentage, validation_percentage)

# Iniciando uma sessao
#import 
import tensorflow as tf
import numpy as np

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.InteractiveSession(config=config)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
# ess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, width * height])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])


# Inicializacao das Variaveis

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolucao e Pooling

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x,ksize=[1, 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='SAME')

# Funcoes auxiliares

def next_batch(data, size):
    global index
    start = index
    index += size
    if index > data[0].shape[0]:
        perm = np.arange(data[0].shape[0])
        np.random.shuffle(perm)
        data[0] = data[0][perm]
        data[1] = data[1][perm]
        start = 0
        index = size
    end = index
    return data[0][start:end], data[1][start:end]


# Primeira Camada Convolucional

W_conv1 = weight_variable([55, 55, 1, 96])
b_conv1 = bias_variable([96])

x_image = tf.reshape(x, [-1, width, height, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 64x64

# Camada Convolucional - Primeira Coluna

W_conv2_1c1 = weight_variable([15, 15, 96, 64])
b_conv2_1c1 = bias_variable([64])

h_conv2_1c1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1c1) + b_conv2_1c1)

# Camada Convolucional - Segunda Coluna 1

W_conv2_2c1 = weight_variable([10, 10, 96, 128])
b_conv2_2c1 = bias_variable([128])

h_conv2_2c1 = tf.nn.relu(conv2d(h_pool1, W_conv2_2c1) + b_conv2_2c1)

# Camada Convolucional - Segunda Coluna 2

W_conv2_2c2 = weight_variable([5, 5, 128, 64])
b_conv2_2c2 = bias_variable([64])

h_conv2_2c2 = tf.nn.relu(conv2d(h_conv2_2c1, W_conv2_2c2) + b_conv2_2c2)

# Camada Convolucional - Terceira Coluna
h_apool2_3c1 = avg_pool_2x2(h_pool1)

W_conv2_3c1 = weight_variable([20, 20, 96, 64])  # 2,2,64,256
b_conv2_3c1 = bias_variable([64])

h_conv2_3c1 = tf.nn.relu(conv2d(h_apool2_3c1, W_conv2_3c1) + b_conv2_3c1)


# Camada Densamente Conectada

h_conv3_avg = tf.concat([h_conv2_1c1, h_conv2_2c2, h_conv2_3c1], 0)
h_apool3 = avg_pool_2x2(h_conv3_avg)

W_fc1 = weight_variable([(int(width / (2 ** n_maxpool)) * int(height / (2 ** n_maxpool)) * 64), 2048])  # 2048
b_fc1 = bias_variable([2048])

h_pool2_flat1 = tf.reshape(h_apool3, [-1, (int(width / (2 ** n_maxpool)) * int(height / (2 ** n_maxpool)) * 64)])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat1, W_fc1) + b_fc1)

# Camada de Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc3 = weight_variable([2048, n_classes])
b_fc3 = bias_variable([n_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

# Treinamento e Avaliacao do Desempenho da Rede Neural

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

for i in range(max_steps):
    batch = next_batch(images["training"], batch_size)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})

        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: images["testing"][0], y_: images["testing"][1], keep_prob: 1.0}))

sess.close()
stop = timeit.default_timer()
print("duration: %.2f min" % ((stop - start) / 60))
