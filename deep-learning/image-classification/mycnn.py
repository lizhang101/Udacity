
# coding: utf-8

# In[1]:

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    print (image_shape)
    x = tf.placeholder(tf.float32, shape=[None]+list(image_shape), name='x')
    return x


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
    return y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    k = tf.placeholder(tf.float32, name='keep_prob')
    return k


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function

    #print (conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    input_shape = x_tensor.get_shape().as_list()
   

    W = tf.Variable(tf.random_normal([conv_ksize[0], conv_ksize[1], input_shape[3], conv_num_outputs], stddev=0.1))
    #b = tf.Variable(tf.random_normal([conv_num_outputs]))
    b = tf.Variable(tf.zeros([conv_num_outputs]))
    
    x = tf.nn.conv2d(x_tensor, W, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    
    x = tf.nn.max_pool(x, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return x 



def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    from functools import reduce
    flat_length = reduce(lambda x,y: x*y, x_tensor.get_shape().as_list()[1:])
    return tf.reshape(x_tensor, [-1, flat_length])



def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    w = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs],stddev=0.1))
    #b = tf.Variable(tf.random_normal([num_outputs]))
    b = tf.Variable(tf.zeros([num_outputs]))
    y = tf.nn.relu(tf.add(tf.matmul(x_tensor, w), b))
    return y


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    w = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs]))
    b = tf.Variable(tf.random_normal([num_outputs]))
    y = tf.add(tf.matmul(x_tensor, w), b)
    return y



def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    #source image: 32x32x3
    
    conv = conv2d_maxpool(x, 64, [5,5], [1,1], [2,2], [2,2])

    #conv = tf.nn.dropout(conv, keep_prob)
    conv = conv2d_maxpool(conv, 128, [5,5],[1,1],[2,2],[2,2])
    conv = conv2d_maxpool(conv, 256, [3,3],[1,1],[2,2],[2,2])
    #conv = tf.nn.dropout(conv, keep_prob)

    conv_out = flatten(conv)
    

    fc = fully_conn(conv_out, 2048)

    #fc = tf.nn.dropout(fc, keep_prob)
    fc = fully_conn(fc, 1024)
    fc = fully_conn(fc, 512)
    fc = tf.nn.dropout(fc, keep_prob)
    
    logits = output(fc, 10)
    
    # TODO: return output
    return logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)



def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss = session.run(cost, feed_dict={x:feature_batch, y:label_batch, keep_prob:1.0})
    train_accu = session.run(accuracy, feed_dict={x:feature_batch, y:label_batch, keep_prob:1.0})
    valid_accu = session.run(accuracy, feed_dict={x:valid_features, y:valid_labels, keep_prob:1.0})
    print (" - Loss: {:>10.4f} Trainning Accuracy: {:.6f} Validation Accuracy: {:.6f}".format(loss, train_accu, valid_accu))
    
# TODO: Tune Parameters

save_model_path = './mycnn'

print('Training...')
def train(model = None, epochs = 1, n_batches = 5, batch_size = 64, keep_probability=0.95):
    with tf.Session() as sess:
        # Initializing the variables
        if model is None:
            sess.run(tf.global_variables_initializer())
        else:
            loader = tf.train.Saver()
            loader.restore(sess, model)
            print ("Restored model:", model)
        
        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
                
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)


# In[1]:
#train(n_batches=1)
train(epochs = 8, n_batches=5, keep_probability=0.7)
#train(epochs = 8, n_batches=5, keep_probability=0.8, model = save_model_path)


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
#get_ipython().magic('matplotlib inline')
#get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        #helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
