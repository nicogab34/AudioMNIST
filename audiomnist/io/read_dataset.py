
import tensorflow as tf

def parse_raw_records(record):
    raw_feature = {
        'data' : tf.FixedLenFeature([8000], tf.float32),
        'digit' : tf.FixedLenFeature([], tf.int64),
        'gender' : tf.FixedLenFeature([], tf.int64),
        'vp' : tf.FixedLenFeature([], tf.int64)
    }
    example = tf.parse_single_example(record, raw_feature)
    example['data'] = tf.reshape(example['data'], (8000,1))
    return example

def parse_spectrogram_records(record):
    spectrogram_feature = {
        'data' : tf.FixedLenFeature([227*227], tf.float32),
        'digit' : tf.FixedLenFeature([], tf.int64),
        'gender' : tf.FixedLenFeature([], tf.int64),
        'vp' : tf.FixedLenFeature([], tf.int64)
    }
    example = tf.parse_single_example(record, spectrogram_feature)
    example['data'] = tf.reshape(example['data'], (227,227,1))
    return example

def load_raw_dataset(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(parse_raw_records)
    return dataset

def load_spectrogram_dataset(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(parse_spectrogram_records)
    return dataset