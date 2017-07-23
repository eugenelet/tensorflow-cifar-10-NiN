import tensorflow as tf
from math import sqrt


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _RESHAPE_SIZE = 4 * 4 * 128

    def put_kernels_on_grid(kernel, pad=1):

        '''Visualize conv. features as an image (mostly for the 1st layer).
        Place kernel into a grid, with some paddings between adjacent filters.
        Args:
          kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
          (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                               User is responsible of how to break into two multiples.
          pad:               number of black pixels around each filter (between them)
        Return:
          Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
        '''

        # get shape of the grid. NumKernels == grid_Y * grid_X
        def factorization(n):
            for i in range(int(sqrt(float(n))), 0, -1):
                if n % i == 0:
                    if i == 1:
                        print('Who would enter a prime number of filters')
                    return (i, int(n / i))

        (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)

        kernel1 = (kernel - x_min) / (x_max - x_min)

        # pad X and Y
        x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

        # X and Y dimensions, w.r.t. padding
        Y = kernel1.get_shape()[0] + 2 * pad
        X = kernel1.get_shape()[1] + 2 * pad

        channels = kernel1.get_shape()[2]

        # put NumKernels to the 1st dimension
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        # organize grid on Y axis
        x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

        # switch X and Y axes
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        # organize grid on X axis
        x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

        # back to normal order (not combining with the next step for clarity)
        x6 = tf.transpose(x5, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels],
        #   where in this case batch_size == 1
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scaling to [0, 255] is not necessary for tensorboard
        return x7

    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    # define functions
    def weight_variable(shape, stddev=0.05):
        initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3x3(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, [None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS])
    y = tf.placeholder(tf.float32, [None, _NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS])

    # conv1 layer
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 192], stddev=0.01)
        b_conv1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
        output = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram('conv_filter', output)
        tf.summary.scalar('conv_filter', tf.nn.zero_fraction(output))
        # MLP-1-1
        with tf.name_scope('mlp_1_1'):
            W_MLP11 = weight_variable([1, 1, 192, 160])
            b_MLP11 = bias_variable([160])
            output = tf.nn.relu(conv2d(output, W_MLP11) + b_MLP11)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # MLP-1-2
        with tf.name_scope('mlp_1_2'):
            W_MLP12 = weight_variable([1, 1, 160, 96])
            b_MLP12 = bias_variable([96])
            output = tf.nn.relu(conv2d(output, W_MLP12) + b_MLP12)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))

        with tf.variable_scope('Visualization'):
            grid = put_kernels_on_grid(W_conv1)
            tf.summary.image('conv1/filters', grid, max_outputs=1)

        # Max pooling
        output = max_pool_3x3(output)
        # dropout
        output = tf.nn.dropout(output, keep_prob)

    # conv2 layer
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 96, 192])
        b_conv2 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
        output = tf.nn.relu(conv2d(output, W_conv2) + b_conv2)
        tf.summary.histogram('conv_filter', output)
        tf.summary.scalar('conv_filter', tf.nn.zero_fraction(output))
        # MLP-2-1
        with tf.name_scope('mlp_2_1'):
            W_MLP21 = weight_variable([1, 1, 192, 192])
            b_MLP21 = bias_variable([192])
            output = tf.nn.relu(conv2d(output, W_MLP21) + b_MLP21)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # MLP-2-2
        with tf.name_scope('mlp_2_2'):
            W_MLP22 = weight_variable([1, 1, 192, 192])
            b_MLP22 = bias_variable([192])
            output = tf.nn.relu(conv2d(output, W_MLP22) + b_MLP22)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # Max pooling
        output = max_pool_3x3(output)
        # dropout
        output = tf.nn.dropout(output, keep_prob)

    # conv3 layer
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 192, 192])
        b_conv3 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
        output = tf.nn.relu(conv2d(output, W_conv3) + b_conv3)
        tf.summary.histogram('conv_filter', output)
        tf.summary.scalar('conv_filter', tf.nn.zero_fraction(output))
        # MLP-2-1
        with tf.name_scope('mlp_3_1'):
            W_MLP31 = weight_variable([1, 1, 192, 192])
            b_MLP31 = bias_variable([192])
            output = tf.nn.relu(conv2d(output, W_MLP31) + b_MLP31)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # MLP-2-2
        with tf.name_scope('mlp_3_2'):
            W_MLP32 = weight_variable([1, 1, 192, 10])
            b_MLP32 = bias_variable([10])
            output = tf.nn.relu(conv2d(output, W_MLP32) + b_MLP32)
            tf.summary.histogram('mlp', output)
            tf.summary.scalar('mlp', tf.nn.zero_fraction(output))
        # global average
        output = tf.nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        # [n_samples, 1, 1, 10] ->> [n_samples, 1*1*10]
    with tf.name_scope('output'):
        output = tf.reshape(output, [-1, 1 * 1 * 10])
        tf.summary.histogram('avg_pool', output)


    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    y_pred_cls = tf.argmax(output, dimension=1)

    return x, y, output, global_step, y_pred_cls, keep_prob
