import tensorflow as tf

files = tf.train.match_filenames_once("./output.tfrecords")
filename_queue = tf.train.string_input_producer(files,shuffle=False)


reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })

decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'],tf.int32)
#pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

images_batch, labels_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000

weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

y = inference(images_batch,weight1,bias1,weight2,bias2)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=labels_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weight1) + regularizer(weight2)
loss = cross_entropy_mean + regularaztion

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            print("After %d training steps ,the loss is %g " % (i, sess.run(loss)))

            sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
