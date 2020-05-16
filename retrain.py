# Code adapted by Daniel Kermany and the Zhang Lab team 2017

# Sample Usage:
# python retrain.py --images /path/to/images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from net import utils
from net import train
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

FLAGS = None

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Create directories to store TensorBoard summaries 
  utils.create_tensorboard_directories(FLAGS.summaries_dir)

  # Set model properties
  model_config = train.get_model_config()

  # Set up the pre-trained graph.
  utils.download_pretrained_weights(model_config["inception_url"], FLAGS.model_dir)
  graph, bottleneck_tensor, resized_image_tensor = train.create_model_graph(
    model_config, FLAGS.model_dir)

  # Look at the folder structure, and create lists of all the images.
  image_lists = utils.create_image_lists(FLAGS.images)
  class_count = len(image_lists.keys())

  if class_count == 0:
    tf.logging.error("No valid folders of images found at " + FLAGS.images)
    return -1
  if class_count == 1:
    tf.logging.error("Only one valid folder of images found at " +
                     FLAGS.images +
                     " - multiple classes are needed for classification.")
    return -1

  # Create output_labels.txt displaying classes being trained
  with gfile.FastGFile(FLAGS.output_labels, "w") as f:
    f.write("\n".join(image_lists.keys()) + "\n")

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph.
    jpeg_data_tensor, decoded_image_tensor = utils.decode_jpeg(
      model_config["input_width"], model_config["input_height"],
      model_config["input_depth"], model_config["input_mean"],
      model_config["input_std"])

    # Store image bottlenecks
    train.store_bottlenecks(
      sess, image_lists, FLAGS.images, FLAGS.bottleneck_dir,
      jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
      bottleneck_tensor)

    # Train newly initialized final layer
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
    final_tensor) = train.train_final_layer(
      len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor,
      model_config["bottleneck_tensor_size"], FLAGS.learning_rate)

    # Create evaluation graph
    evaluation_step, prediction, probability = train.create_evaluation_graph(
      final_tensor, ground_truth_input)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/train",
                                           sess.graph)

    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + "/validation")

    # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    best_acc = 0.0

    since = time.time()
    for i in range(FLAGS.training_steps):
      (train_bottlenecks, train_ground_truth, _) = train.get_batch_of_stored_bottlenecks(
        sess, image_lists, FLAGS.train_batch_size, "training",
        FLAGS.bottleneck_dir, FLAGS.images, jpeg_data_tensor,
        decoded_image_tensor, resized_image_tensor, bottleneck_tensor)
      # Run a training step 
      train_summary, _ = sess.run(
        [merged, train_step],
        feed_dict={bottleneck_input: train_bottlenecks,
                   ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Show evaluation based on specified frequency
      final_step = (i + 1 == FLAGS.training_steps)
      if (i % FLAGS.eval_frequency) == 0 or final_step:
        # Run evaluation step on training bottlenecks  
        train_accuracy, cross_entropy_value = sess.run(
          [evaluation_step, cross_entropy],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
        # Fetch validation bottlenecks for evaluation
        validation_bottlenecks, validation_ground_truth, _ = (
          train.get_batch_of_stored_bottlenecks(
            sess, image_lists, FLAGS.validation_batch_size, "validation",
            FLAGS.bottleneck_dir, FLAGS.images, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor))
        # Run evaluation step on validation bottlenecks
        validation_summary, validation_accuracy = sess.run(
          [merged, evaluation_step],
          feed_dict={bottleneck_input: validation_bottlenecks,
                     ground_truth_input: validation_ground_truth})
        # Save best accuracy and store model to disk
        if validation_accuracy > best_acc:
          best_acc = validation_accuracy
          utils.save_graph_to_file(sess, graph, FLAGS.output_graph, FLAGS.final_tensor_name)
            
        validation_writer.add_summary(validation_summary, i)

        tf.logging.info("Step {}: loss = {} train acc = {} val acc = {}". format(
          i, cross_entropy_value, train_accuracy, validation_accuracy))

    # Training complete. Run final evaluation on test set
    # Fetch test bottlenecks for evaluation
    test_bottlenecks, test_ground_truth, test_filenames = (
      train.get_batch_of_stored_bottlenecks(
        sess, image_lists, FLAGS.test_batch_size, "testing",
        FLAGS.bottleneck_dir, FLAGS.images, jpeg_data_tensor,
        decoded_image_tensor, resized_image_tensor, bottleneck_tensor))
    # Run evaluation step on test bottlenecks
    test_accuracy, predictions, probabilities = sess.run(
      [evaluation_step, prediction, probability],
      feed_dict={bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth})
    tf.logging.info("Best validation accuracy = {}".format(best_acc * 100))
    tf.logging.info("Final test accuracy =  {}".format(test_accuracy * 100))

  time_elapsed = time.time() - since

  predictions = np.argmax(probabilities, axis=1)
  labels = np.argmax(test_ground_truth, axis=1)
  print("Total Model Runtime: {}min, {:0.2f}sec".format(int(time_elapsed // 60), time_elapsed % 60))
    
  # roc_labels = [0 if label in [LIST_OF_POS_IDX] else 1 for label in labels]
  # pos_probs = probabilities[:, [LIST_OF_POS_IDX]]
  # roc_probs = np.ndarray.sum(pos_probs, axis=1)
  # auc = utils.generate_roc(roc_labels, roc_probs, pos_label = 0)
  # print("Final Model AUC: {:0.2f}%".format(auc * 100))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--images",
      type=str,
      default="",
      help="Path to folder containing subdirectories of the training \
            categories (filesnames all CAPS)"
  )
  parser.add_argument(
      "--output_graph",
      type=str,
      default="/tmp/retrained_graph.pb",
      help="Output directory to save the trained graph."
  )
  parser.add_argument(
      "--output_labels",
      type=str,
      default="/tmp/output_labels.txt",
      help="Directory in which to save the labels."
  )
  parser.add_argument(
      "--summaries_dir",
      type=str,
      default="/tmp/retrain_logs",
      help="Path to save summary logs for TensorBoard."
  )
  parser.add_argument(
      "--training_steps",
      type=int,
      default=4000,
      help="How many training steps to run before ending."
  )
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Set learning rate"
  )
  parser.add_argument(
      "--eval_frequency",
      type=int,
      default=10,
      help="How often to evaluate the training results."
  )
  parser.add_argument(
      "--train_batch_size",
      type=int,
      default=256,
      help="How many images to train on at a time."
  )
  parser.add_argument(
      "--test_batch_size",
      type=int,
      default=-1,
      help="Number of images from test set to test on. Value of -1 will \
            cause entire directory to be used. Since it is used only \
            once, -1 will work in most cases."
  )
  parser.add_argument(
      "--validation_batch_size",
      type=int,
      default=-1,
      help="Number of images from validation set to validate on. Value of \
            -1 will cause entire directory to be used. Large batch sizes \
            may slow down training size it is performed frequently."
  )
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/tmp/imagenet",
      help="Path to pretrained weights"
  )
  parser.add_argument(
      "--bottleneck_dir",
      type=str,
      default="/tmp/bottleneck",
      help="Path to store bottleneck layer values."
  )
  parser.add_argument(
      "--final_tensor_name",
      type=str,
      default="final_result",
      help="The name of the output classification layer in the retrained graph."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
