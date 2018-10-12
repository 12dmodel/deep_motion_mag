import argparse
import os
import glob
import sys
import numpy as np
from tqdm import tqdm
import cv2
import tensorflow as tf
import json

FLAGS = None

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_dataset(data_dir, out_name, color=False):
    # Open a TFRRecordWriter
    filename = os.path.join(out_name)
    writeOpts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writeOpts = tf.python_io.TFRecordOptions()
    writer = tf.python_io.TFRecordWriter(filename, options=writeOpts)

    # Load each data sample (image_a, image_b, flow) and write it to the TFRecord
    for f in tqdm(glob.glob(os.path.join(data_dir, 'frameA', '*.png'))):
        f = os.path.basename(f)
        image_a_path = os.path.join(data_dir, 'frameA', f)
        image_b_path = os.path.join(data_dir, 'frameB', f)
        image_c_path = os.path.join(data_dir, 'frameC', f)
        flow_path = os.path.join(data_dir, 'amplified', f)
        f, _ = os.path.splitext(f)
        meta_path = os.path.join(data_dir, 'meta', f + '.json')

        if color:
            flag = cv2.IMREAD_COLOR
        else:
            flag = cv2.IMREAD_GRAYSCALE
        image_a = cv2.imread(image_a_path, flags=flag).astype('uint8')
        image_b = cv2.imread(image_b_path, flags=flag).astype('uint8')
        image_c = cv2.imread(image_c_path, flags=flag).astype('uint8')
        flow = cv2.imread(flow_path, flags=flag).astype('uint8')

        if color:
            image_a = cv2.cvtColor(image_a, code=cv2.COLOR_BGR2RGB)
            image_b = cv2.cvtColor(image_b, code=cv2.COLOR_BGR2RGB)
            image_c = cv2.cvtColor(image_c, code=cv2.COLOR_BGR2RGB)
            flow = cv2.cvtColor(flow, code=cv2.COLOR_BGR2RGB)

        amplification_factor = json.load(open(meta_path))['amplification_factor']
        # Scale from [0, 255] -> [0.0, 1.0]
        # image_a = image_a / 255.0
        # image_b = image_b / 255.0
        # flow = flow / 255.0

        image_a_raw = image_a.tostring()
        image_b_raw = image_b.tostring()
        image_c_raw = image_c.tostring()
        flow_raw = flow.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'frameA': _bytes_feature(image_a_raw),
            'frameB': _bytes_feature(image_b_raw),
            'frameC': _bytes_feature(image_c_raw),
            'amplified': _bytes_feature(flow_raw),
            'amplification_factor': _float_feature(amplification_factor),
            }))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    # Convert the train and val datasets into .tfrecords format
    convert_dataset(os.path.join(FLAGS.data_dir, 'train'), os.path.join(FLAGS.out, 'train.tfrecords'), FLAGS.color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory that includes all .png files in the dataset'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Directory for output .tfrecords files'
    )
    parser.add_argument('--color', action='store_true', help='Whether to store image as color.')
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('data_dir must exist and be a directory')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out must exist and be a directory')
    main()
