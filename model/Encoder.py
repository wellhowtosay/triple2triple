import tensorflow as tf
import model.Transformer as Transformer

config = {
    "LSTM": {

    }
}
encoders = {"LSTM": tf.keras.layers.LSTM, "Transfomer": Transformer}


def get_encoder(name, config):
    return encoders[name](config)
