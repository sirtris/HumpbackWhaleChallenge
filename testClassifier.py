from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import os

tf.logging.set_verbosity(tf.logging.INFO)

#Source: https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) #replace by final image size
    """#Old convolutional layers
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
    # Dense Layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    """
    flat_input = tf.reshape(input_layer, [-1,28*28]) #TODO: replace by final dimensions

    lin1 = tf.contrib.layers.fully_connected(
        inputs=flat_input,
        num_outputs=28 *28, #TODO: replace by final dimensions
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer()
    )

    lin2 = tf.contrib.layers.fully_connected(
        inputs=lin1,
        num_outputs=28*28, #TODO: replace by final dimensions
        activation_fn=tf.nn.relu,
        biases_initializer=tf.zeros_initializer()
    )

    dropout = tf.layers.dropout(
        inputs=lin2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=200) #Set to nr of classes

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def augment_images(files, directory):
    idealWidth = 1050
    idealHeight = 600
    for image in files:
        fpath = directory + image
        img = Image.open(fpath)

        # Augment images
        img = img.resize((idealWidth, idealHeight), Image.NEAREST)
        img = img.convert("L")
        #Other transformations?

        img.save('altered/' + image)

def image_to_float(im):
    arr = np.array(im)
    return [[float(x) for x in row[0:28]] for row in arr[0:28]]

def getWhales():
    dir = os.path.dirname(__file__)
    train = os.listdir('data/train')
    test = os.listdir('data/test')
    train_dir = 'data/train/'
    test_dir = 'data/test/'

    #Import the whale data
    #augment_images(train,train_dir) #RESIZE IMAGES
    train_df = pd.read_csv('./data/train.csv')
    ids = train_df['Id'].values.tolist()
    train_data = np.array([image_to_float(Image.open('altered/' + x)) for x in train[0:100]])
    eval_data = np.array([image_to_float(Image.open('altered/' + x)) for x in train[100:200]])

    #Import the data labels and convert them to integer
    train_labels = ids[0:100]
    eval_labels = ids[100:200]
    unique_labels = list(set(np.concatenate((train_labels,eval_labels))))
    n_classes = len(unique_labels)
    train_labels_int = np.array([unique_labels.index(x) for x in train_labels]) #LABELS CONVERTED TO INTEGER SINCE STRINGS ARE NOT ACCEPTED
    eval_labels_int = np.array([unique_labels.index(x) for x in eval_labels])
    return train_data,train_labels_int,eval_data,eval_labels_int

def main(unused_argv):
    # Load training and eval data
    """mnist = tf.contrib.learn.datasets.load_dataset("mnist") #TODO: Replice by import function of cropped whale data
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32) """
    train_data,train_labels,eval_data,eval_labels = getWhales()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/whale_linear")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()