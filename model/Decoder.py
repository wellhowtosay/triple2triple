# R.I.P
# class AttentionLSTMCell(tf.keras.layers.LSTMCell):
#     def __init__(self, **cell_config):
#         super(AttentionLSTMCell, self).__init__(**cell_config)
#         self.attention = tf.keras.layers.Attention()
#
#     def set_value_seq(self, value_seq):
#         self.value_seq = value_seq
#
#     def build(self, inputs_shape):
#         self.qurey_dense = tf.keras.layers.Dense(inputs_shape[-1])
#         super().build(inputs_shape)
#
#     def call(self, inputs, states, training=None):
#         # states[0] is the previous output,states[1] is the hidden state
#         query_seq = self.qurey_dense(tf.expand_dims(states[0], 1))
#         inputs = tf.squeeze(self.attention([query_seq, self.value_seq], [None, self.value_seq._keras_mask]), 1)
#         return super().call(inputs, states, training)
#
#
# class AttentionRNN(tf.keras.layers.RNN):
#     def __init__(self,
#                  cell,
#                  **kwargs):
#         super(AttentionRNN, self).__init__(cell,
#                                            **kwargs)
#
#     def call(self, inputs,
#              mask=None,
#              training=None,
#              initial_state=None,
#              constants=None):
#         self.cell.set_value_seq(inputs)
#         return super().call(inputs,
#                             mask=mask,
#                             training=training,
#                             initial_state=initial_state,
#                             constants=constants)

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from absl.testing import parameterized

configs = {
    "LSTM":
        {
            "AttentionMechanism": [tfa.seq2seq.BahdanauAttention, tfa.seq2seq.LuongAttention],
            "num_units": [10, 192, 256],  # units of LSTM = attention_layer_size
            "decoder_embedding_input_dim": [3]
        }
}


def get_config(name, index):
    config = {}
    for (i, k) in enumerate(configs[name]):
        config[k] = configs[name][k][index[i]]
    return config


class BahdanauAttentionLSTMDecoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BahdanauAttentionLSTMDecoder, self).__init__()
        self.num_units = config["num_units"]
        self.decoder_embedding_input_dim = config["decoder_embedding_input_dim"]
        self.cell = tf.keras.layers.LSTMCell(self.num_units)
        # sampler起到argmax的作用
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.attention_mechanism = tfa.seq2seq.BahdanauAttention(units=self.num_units)
        self.attention_cell = tfa.seq2seq.AttentionWrapper(self.cell, self.attention_mechanism,
                                                           attention_layer_size=self.num_units)  # attention_layer_size
        self.output_layer = tf.keras.layers.Dense(self.decoder_embedding_input_dim)  # 逻辑上是分类问题的类别数
        self.decoder = tfa.seq2seq.BasicDecoder(cell=self.attention_cell, sampler=self.sampler,
                                                output_layer=self.output_layer)

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        # 这个应该是每个batch都换一次，否则某个特定random会变成超参
        self.cell_state = [tf.random.normal(shape=(self.batch_size, self.num_units)),
                           tf.random.normal(shape=(self.batch_size, self.num_units))]
        self.sequence_length = input_shape[1][1]

    # inputs=(encoder_output,decoder_embedding_output,encoder_state[''])
    def call(self, inputs):
        memory_sequence_length = tf.constant(tf.reduce_sum(tf.cast(inputs[0]._keras_mask, tf.int8), -1))
        self.attention_mechanism.setup_memory(memory=inputs[0], memory_sequence_length=memory_sequence_length)
        decoder_initial_state = self.attention_cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        # cell_state[0]是用于第一个查询
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=self.cell_state)
        print(decoder_initial_state)
        outputs, _, _ = self.decoder(inputs=inputs[1], initial_state=decoder_initial_state,
                                     sequence_length=tf.constant([self.sequence_length]))
        return outputs


def get_decoder(name, config_index):
    config = get_config(name, config_index)
    if name == "LSTM":
        return BahdanauAttentionLSTMDecoder(config)


de = BahdanauAttentionLSTMDecoder(get_config("LSTM", index=[0, 0, 0]))
input = np.array([[2, 1, 2, 1, 1, 0]])
decoder_input = np.array([[2, 1, 1, 2, 1,  2, 1, 0, 0]])
embedding = tf.keras.layers.Embedding(3, 10, mask_zero=True)
decoder_embedding = tf.keras.layers.Embedding(3, 10)
embedding_output = embedding(input)
decoder_embedding_output = decoder_embedding(decoder_input)
encoder = tf.keras.layers.LSTM(units=10, return_sequences=True)
encoder_output = encoder(embedding_output)
decoder_output = de((encoder_output, decoder_embedding_output))
print(decoder_output.rnn_output)
print(decoder_output.sample_id)

