import sys
import os
from glob import glob
from six.moves import urllib
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow as tf
import tarfile

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# Create directory if it does not already exist
def create_directory(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Return a sorted list of image files at image_dir
def get_image_files(image_dir):
  fs = glob("{}/*.jpeg".format(image_dir))
  fs = [os.path.basename(filename) for filename in fs]
  return sorted(fs)

# Generates ROC plot and returns AUC using sklearn
def generate_roc(y_test, y_score, pos_label = 0):
  fpr, tpr, _ = roc_curve(y_test, y_score, pos_label = pos_label)
  roc_auc = auc(fpr, tpr)
  plt.figure()
  plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
  plt.plot([0, 1], [0, 1], "k--")
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver operating characteristic curve")
  plt.show()
  return roc_auc

# Download inception model if not already at 'inception_url'
def download_pretrained_weights(inception_url, dest_dir):
  create_directory(dest_dir)      
  filename = inception_url.split("/")[-1]
  filepath = os.path.join(dest_dir, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:0.1f}".format(
        filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(inception_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info("Successfully downloaded", filename, statinfo.st_size,
                                                                                "bytes.")
    tarfile.open(filepath, "r:gz").extractall(dest_dir)


# Save retrained model (.pb) to' output_file'
def save_graph_to_file(sess, graph, output_file, final_tensor_name):
  output_graph_def = graph_util.convert_variables_to_constants(
    sess, graph.as_graph_def(), [final_tensor_name])
  with gfile.FastGFile(output_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())


# Take in image directory and return a dictionary containing images split
# into training, testing, and validation split into each label
def create_image_lists(image_dir):
  result = {}

  training_images = []
  testing_images = []
  validation_images = []

  for category in ["train", "test", "val"]:
    category_path = os.path.join(image_dir, category)
    try:
      bins = next(os.walk(category_path))[1]
    except StopIteration:
      sys.exit("ERROR: Missing either train/test/val folders in image_dir")
    for diagnosis in bins:
      bin_path = os.path.join(category_path, diagnosis)
      if category == "train":
        training_images.append(get_image_files(bin_path))
      if category == "test":
        testing_images.append(get_image_files(bin_path))
      if category == "val":
        validation_images.append(get_image_files(bin_path))

  for diagnosis in bins:
    result[diagnosis] = {
      "training": training_images[bins.index(diagnosis)],
      "testing": testing_images[bins.index(diagnosis)],
      "validation": validation_images[bins.index(diagnosis)],
    }
  return result

# Return a path to an image with the given label at the given index
def get_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
    tf.logging.fatal("Label does not exist %s.", label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal("Category does not exist %s.", category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal("Label %s has no images in the category %s.", label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]

  if("train" in category):
    full_path = os.path.join(image_dir, "train", label_name.upper(), base_name)
  elif("test" in category):
    full_path = os.path.join(image_dir, "test", label_name.upper(), base_name)
  elif("val" in category):
    full_path = os.path.join(image_dir, "val", label_name.upper(), base_name)

  return full_path

# Return path to bottleneck for a given label and a given index
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
  return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + "_inception_v3.txt"

# Create directories for TensorBoard summaries
def create_tensorboard_directories(summaries_dir):
  if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
  tf.gfile.MakeDirs(summaries_dir)

# Returns tensors to feed jpeg data into
def decode_jpeg(input_width, input_height, input_depth, input_mean,input_std):
  jpeg_data = tf.placeholder(tf.string, name="DecodeJPEGInput")
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image
