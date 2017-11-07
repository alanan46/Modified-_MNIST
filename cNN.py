#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for M-MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)
label_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25,
 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1] new [batch_size, 64, 64, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32] new [batch_size, 64, 64, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32] new [batch_size, 64, 64, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32] new [batch_size, 32, 32,64]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32] new [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64] new [batch_size, 32, 32, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]  new [batch_size, 32, 32, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]  new [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#conv3
#Input [batch_size,16,16,64]
#Output [batch_size 16,16,96]
  conv3= tf.layers.conv2d(
      inputs=pool2,
      filters=96,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)


      #pool3
      #Input [batch_size 16,16,96]
      #output [batch_size 8,8,96]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64] new [batch_size, 8, 8, 96]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]  new [batch_size, 16, 16, 96]
  pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 96])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64] new [batch_size, 16, 16, 192]
  # Output Tensor Shape: [batch_size, 1024]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
  dense3= tf.layers.dense(inputs=dense2, units=1024, activation=tf.nn.relu)
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense3, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10] [batch_size, 40]
  logits = tf.layers.dense(inputs=dropout, units=40)
  pred_class=tf.argmax(input=logits, axis=1)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": pred_class,
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=40)
  with tf.name_scope("Evaluating") as scope:
    correct_prediction = tf.equal(pred_class, tf.argmax(onehot_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  with tf.name_scope("Loss") as scope:
    loss_log = tf.summary.scalar("loss ", loss)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
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



def main(unused_argv):

  # Load training and eval data
  # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  sess=tf.Session()
  x = np.loadtxt("half2_train_x.csv", delimiter=",") # load from text
  y = np.loadtxt("half2_train_y.csv", delimiter=",")
  #x = x.reshape(-1, 64, 64) # reshape
  y = y.reshape(-1)
  y_class=[]
  for element in y:
      y_class.append(label_class.index(element))
  y_class=np.array(y_class)
  y_class=y_class.astype(np.int32,copy=False)
  x = x.astype(np.float32, copy=False)
  #y = y.astype(np.int32, copy=False)

  train_data, eval_data, train_labels, eval_labels = train_test_split(x, y_class, test_size=0.1, random_state=42)
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./mm_mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
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
      steps=1000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  #tf.summary.scalar("cross_accuracy", eval_results)
  merged = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('./tensorflow/log3', sess.graph)


if __name__ == "__main__":
  tf.app.run()
