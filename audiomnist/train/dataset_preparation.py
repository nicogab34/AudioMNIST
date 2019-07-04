import tensorflow as tf
from .splits import splits

def prepare_datasets_digit_cls(dataset, batch_size):
    datasets = {}
    sizes = {}

    datasets["train"] = dataset.filter(split('digit', 'train')) \
        .map(make_tuple_cls) \
        .shuffle(10000, seed=42).batch(batch_size) \
        .repeat()

    sizes["train"] = len(splits['digit']['train'][0])*500

    datasets["validation"] = dataset.filter(split('digit', 'validate')) \
        .map(make_tuple_cls) \
        .shuffle(10000, seed=42).batch(batch_size) \
        .repeat()

    sizes["validation"]  = len(splits['digit']['validate'][0])*500

    datasets["test"] = dataset.filter(split('digit', 'test')) \
        .map(make_tuple_cls) \
        .shuffle(10000, seed=42).batch(batch_size)

    sizes["test"] = len(splits['digit']['test'][0])*500

    return datasets, sizes

def make_tuple_cls(record):
    return (record['data'], tf.one_hot(record['digit'], 10))

def split(task, type):
    set = tf.constant(list(splits[task][type][0]), dtype=tf.int64)
    return lambda record: tf.reduce_any(tf.equal(record['vp'], set))

def prepare_datasets_autoencoder(dataset):
    pass
