import os
import sys
import cv2
import cnn_vgg16

import numpy as np
import tensorflow as tf

from multiprocessing.dummy import Pool as ThreadPool

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIRECTORY = os.path.join('DATA', 'Img')
PRE_PROCESSED_DIRECTORY = os.path.join('DATA', 'Img_Pre')
# RESCALE_SIZE = [224, 224]
SIZE = 224
SAMPLE_CATEGORY_IMG_FILE_TRAIN = "sample_category_img_train.txt"
SAMPLE_CATEGORY_IMG_FILE_VALIDATION = "sample_category_img_validation.txt"

MODEL_DIR = 'top_bottom_convnet_model'

N_THREADS = 8

def pre_process_images():
    if not os.path.exists(PRE_PROCESSED_DIRECTORY):
        os.mkdir(PRE_PROCESSED_DIRECTORY)

    paths = []
    for filename in (SAMPLE_CATEGORY_IMG_FILE_TRAIN, SAMPLE_CATEGORY_IMG_FILE_VALIDATION):
        with open(filename, 'r') as tsv_f:
            for line in tsv_f.readlines():
                imgfile, _ = line.split('\t')

                original_file = os.path.join(DATA_DIRECTORY, imgfile)
                compressed_file = os.path.join(PRE_PROCESSED_DIRECTORY, imgfile)
                paths.append((original_file, compressed_file))

    def _make_compressed(paths):
        import ipdb; ipdb.set_trace()
        original, compressed = paths
        original_img = cv2.imread(original)
        compressed_img = cv2.resize(original_img, (SIZE, SIZE))
        return compressed_img
        # return cv2.imwrite(compressed, compressed_img)

    # pool = ThreadPool(N_THREADS)
    # results = pool.map(_make_compressed, paths)
    res = [_make_compressed(path) for path in paths]
    import ipdb; ipdb.set_trace()
    return results

def parse_images(filename):
    image_reader = tf.WholeFileReader()

    images = []
    labels = []
    with open(filename, 'r') as tsv_f:
        for line in tsv_f.readlines():
            imgfile, cat = line.split('\t')
            labels.append(int(cat) - 1)

            full_img_path = os.path.join(PRE_PROCESSED_DIRECTORY, imgfile)
            img = cv2.imread(full_img_path)
            images.append(img)
            # imgfile = os.path.join(DATA_DIRECTORY, imgfile)
            # _, image_f = image_reader.read(tf.train.string_input_producer([imgfile]))
            # image = tf.image.decode_jpeg(image_f, channels=3)
            # resized_image = tf.image.resize_images(image, RESCALE_SIZE)
            # images.append(resized_image)

    # return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)
    # return tf.train.batch(images, 100), tf.train.batch(labels, 100)
    ds = tf.data.Dataset.from_tensor_slices(images, labels)
    iterator = ds.shuffle(len(images) + 1).batch(10).make_one_shot_iterator()
    return iterator.get_next()

def main(argv):
    # Create the Estimator
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    top_bottom_classifier = tf.estimator.Estimator(model_fn=cnn_vgg16.vgg16, model_dir=MODEL_DIR)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

    if '--pre-process' in argv:
        pre_process_images()

    if '--train' in argv:
        top_bottom_classifier.train(
                input_fn=lambda: parse_images(SAMPLE_CATEGORY_IMG_FILE_TRAIN),
                steps=20000,
                hooks=[logging_hook])

    eval_results = top_bottom_classifier.evaluate(
            input_fn=lambda: parse_images(SAMPLE_CATEGORY_IMG_FILE_VALIDATION))

    print(eval_results)

if __name__=="__main__":
    tf.app.run()
