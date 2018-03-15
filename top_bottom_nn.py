import os
import sys
import cnn_vgg16

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIRECTORY = os.path.join('DATA', 'Img')
RESCALE_SIZE = [224, 224]
SAMPLE_CATEGORY_IMG_FILE_TRAIN = "sample_category_img_train.txt"
SAMPLE_CATEGORY_IMG_FILE_VALIDATION = "sample_category_img_validation.txt"

BATCH_SIZE = 20

def read_img_file(filename):
    with open(filename, 'r') as tsv_f:
        rows = (line.split() for line in tsv_f.readlines())
        return [(img, int(cat)-1) for img, cat in rows]

def process_images(table, batch=None):
    image_reader = tf.WholeFileReader()

    images = []
    labels = []

    sub_table = table[batch:batch+BATCH_SIZE] if batch is not None else table
    for imgfile, cat in sub_table:
        full_path = os.path.join(DATA_DIRECTORY, imgfile)
        _, image_f = image_reader.read(tf.train.string_input_producer([full_path]))
        image = tf.image.decode_jpeg(image_f, channels=3)
        resized_image = tf.image.resize_images(image, RESCALE_SIZE)
        images.append(resized_image)

        labels.append(cat)

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

def main(argv):
    top_bottom_classifier = tf.estimator.Estimator(
            model_fn=cnn_vgg16.vgg16,
            model_dir="/tmp/top_bottom_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

    if len(argv) > 1 and argv[1] == '-t':
        training_table = read_img_file(SAMPLE_CATEGORY_IMG_FILE_TRAIN)

        batch = 0
        while batch < len(training_table):
            print("TRAINING batch %d" % batch)
            top_bottom_classifier.train(
                    input_fn=lambda: process_images(training_table, batch),
                    hooks=[logging_hook])

    evaluation_table = read_img_file(SAMPLE_CATEGORY_IMG_FILE_VALIDATION)

    eval_results = top_bottom_classifier.evaluate(
            input_fn=lambda: process_images(evaluation_table))

    print(eval_results)

if __name__=="__main__":
    tf.app.run()
