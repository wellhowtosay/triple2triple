import tensorflow as tf
import numpy as np
import argparse

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", "./data/SAOKE_DATA.npz",
    "The input data dir")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", "./result/"
                       "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "train_epoch", 40,
    "There are five kinds of factor, list as cEXT,cNEU,cAGR,cCON,cOPN")

data = np.load("data/SAOKE_DATA.npz", allow_pickle=True)


def train_data_generator():
    for i in range(data['knowledge_train'].size):
        yield (data['knowledge_train'][i], data['natural_train'][i])


ds = tf.data.Dataset.from_generator(
    train_data_generator, (tf.int64, tf.int64), (tf.TensorShape([None]), tf.TensorShape([None])))
ds = ds.shuffle(1000)
padded_ds = ds.padded_batch(
    batch_size=10,
    padded_shapes=ds.output_shapes,
    drop_remainder=False
)
for i in padded_ds.take(5):
    print(i)
