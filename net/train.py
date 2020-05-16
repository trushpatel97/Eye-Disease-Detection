from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from net import utils
import tensorflow as tf
import random
import numpy as np
from tensorflow.python.platform import gfile

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1

# Return dict of model info
def get_model_config():
  inception_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
  bottleneck_tensor_name = "pool_3/_reshape:0"
  bottleneck_tensor_size = 2048
  input_width = 299 
  input_height = 299
  input_depth = 3
  resized_input_tensor_name = "Mul:0"
  model_file_name = "classify_image_graph_def.pb"
  input_mean = 128
  input_std = 128

  return {
      "inception_url": inception_url,
      "bottleneck_tensor_name": bottleneck_tensor_name,
      "bottleneck_tensor_size": bottleneck_tensor_size,
      "input_width": input_width,
      "input_height": input_height,
      "input_depth": input_depth,
      "resized_input_tensor_name": resized_input_tensor_name,
      "model_file_name": model_file_name,
      "input_mean": input_mean,
      "input_std": input_std,
  }

# Calculate bottleneck values for image (if not exists)
def get_bottleneck(sess, image_lists, label_name, index, image_dir,
                   category, bottleneck_dir, jpeg_data_tensor,
                   decoded_image_tensor, resized_input_tensor,
                   bottleneck_tensor):
  label_lists = image_lists[label_name]
  bottleneck_path = utils.get_bottleneck_path(image_lists, label_name, index,
                                              bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck(
      bottleneck_path, image_lists, label_name, index,
      image_dir, category, sess, jpeg_data_tensor,
      decoded_image_tensor, resized_input_tensor,
      bottleneck_tensor)

  with open(bottleneck_path, "r") as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  
  # if loaded or created bottleneck is invalid, recreate
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(",")]
  except ValueError:
    tf.logging.warning("Error reading bottleneck, recreating bottleneck")
    create_bottleneck(
      bottleneck_path, image_lists, label_name, index,
      image_dir, category, sess, jpeg_data_tensor,
      decoded_image_tensor, resized_input_tensor,
      bottleneck_tensor)

    with open(bottleneck_path, "r") as bottleneck_file:
      bottleneck_string = bottleneck_file.read()

    bottleneck_values = [float(x) for x in bottleneck_string.split(",")]
  return bottleneck_values

# Run inference on an image to generate bottleneck values
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):

  # Preprocess input images 
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # Run preprocessed images through the network.
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


# Creates a bottleneck file
def create_bottleneck(bottleneck_path, image_lists, label_name, index,
                      image_dir, category, sess, jpeg_data_tensor,
                      decoded_image_tensor, resized_input_tensor,
                      bottleneck_tensor):
  tf.logging.info("Creating Bottleneck at {}".format(bottleneck_path))
  image_path = utils.get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal("File does not exist {}".format(image_path))
  image_data = gfile.FastGFile(image_path, "rb").read()
  try:
    bottleneck_values = run_bottleneck_on_image(
      sess, image_data, jpeg_data_tensor, decoded_image_tensor,
      resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError("Error bottlenecking {}\n{}".format(image_path, str(e)))
 
  bottleneck_string = ",".join(str(x) for x in bottleneck_values)

  bottleneck_directory = "/".join(bottleneck_path.split("/")[:-1])

  utils.create_directory(bottleneck_directory)

  with open(bottleneck_path, "w") as bottleneck_file:
    bottleneck_file.write(bottleneck_string)

# Store bottleneck values
def store_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor):
  num_bottlenecks = 0
  utils.create_directory(bottleneck_dir)

  for label_name, label_lists in image_lists.items():
    for category in ["training", "testing", "validation"]:
      category_list = label_lists[category]
      for index in range(len(category_list)):
        get_bottleneck(
          sess, image_lists, label_name, index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor)

        num_bottlenecks += 1
        if num_bottlenecks % 100 == 0:
          tf.logging.info("{} bottleneck files created.".format(num_bottlenecks))

# Creates a bottleneck file
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  tf.logging.info("Creating Bottleneck at {}".format(bottleneck_path))
  image_path = utils.get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal("File does not exist {}".format(image_path))
  image_data = gfile.FastGFile(image_path, "rb").read()
  try:
    bottleneck_values = run_bottleneck_on_image(
      sess, image_data, jpeg_data_tensor, decoded_image_tensor,
      resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError("Error bottlenecking {}\n{}".format(image_path, str(e)))
  
  bottleneck_string = ",".join(str(x) for x in bottleneck_values)

  bottleneck_directory = "/".join(bottleneck_path.split("/")[:-1])

  utils.create_directory(bottleneck_directory)

  with open(bottleneck_path, "w") as bottleneck_file:
    bottleneck_file.write(bottleneck_string)

# Add a newly initialized FC layer and softmax layer for training
def train_final_layer(class_count, final_tensor_name, bottleneck_tensor,
                      bottleneck_tensor_size, learning_rate):
  with tf.name_scope("input"):
    bottleneck_input = tf.placeholder_with_default(
      bottleneck_tensor,
      shape=[None, bottleneck_tensor_size],
      name="BottleneckInputPlaceholder")

    ground_truth_input = tf.placeholder(
      tf.float32,
      [None, class_count],
      name="GroundTruthInput")

  with tf.name_scope("final_training_layers"):
    with tf.name_scope("weights"):
      # Initialize random weights
      initial_value = tf.truncated_normal(
        [bottleneck_tensor_size, class_count], stddev=0.001)

      layer_weights = tf.Variable(initial_value, name="final_weights")

      add_variable_summaries(layer_weights)
    with tf.name_scope("biases"):
      layer_biases = tf.Variable(tf.zeros([class_count]), name="final_biases")
      add_variable_summaries(layer_biases)
    with tf.name_scope("WXplusb"):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram("pre_activations", logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram("activations", final_tensor)

  with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope("total"):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar("cross_entropy", cross_entropy_mean)

  with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)

# Add summaries to a tensor for TensorBoard
def add_variable_summaries(variable):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(variable)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(variable))
    tf.summary.scalar('min', tf.reduce_min(variable))
    tf.summary.histogram('histogram', variable)

# Returns a batch of the bottlenecks from storage
def get_batch_of_stored_bottlenecks(sess, image_lists, batch_size, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if batch_size >= 0:
    # Retrieve a random sample of bottlenecks.
    for i in range(batch_size):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = utils.get_image_path(
        image_lists, label_name, image_index,
        image_dir, category)
      bottleneck = get_bottleneck(
        sess, image_lists, label_name, image_index, image_dir, category,
        bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks
    # Used for validation set mainly
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = utils.get_image_path(
          image_lists, label_name, image_index,
          image_dir, category)
        bottleneck = get_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames

# Create evaluation graph
def create_evaluation_graph(result_tensor, ground_truth_tensor):
  with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
      prediction = tf.argmax(result_tensor, 1)
      probability = result_tensor[:, :]

      correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope("accuracy"):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar("accuracy", evaluation_step)
  return evaluation_step, prediction, probability

# Return graph of network and bottleneck tensor
def create_model_graph(model_info, model_dir):
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(model_dir, model_info["model_file_name"])
    with gfile.FastGFile(model_path, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
        graph_def,
        name="",
        return_elements=[
          model_info["bottleneck_tensor_name"],
          model_info["resized_input_tensor_name"],
        ]))
  return graph, bottleneck_tensor, resized_input_tensor
