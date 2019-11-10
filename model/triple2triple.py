import tensorflow as tf
import model.Encoder as Encoder
import model.Decoder as Decoder


class Seq2Seq(tf.keras.Model):
    def __init__(self, input_embedding, encoder, decoder, output_embedding):
        super(Seq2Seq, self).__init__()
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.Embedding(input_dim=input_embedding["input_dim"],
                                                   output_dim=input_embedding["output_dim"],
                                                   mask_zero=input_embedding["mask_zero"]))
        self.encoder.add(Encoder.get_encoder(encoder["name"], decoder["config"]))
        self.decoder = Decoder.get_decoder(decoder["name"], decoder["config"])
        self.decoder_embedding = (tf.keras.layers.Embedding(input_dim=output_embedding["input_dim"],
                                                            output_dim=output_embedding["output_dim"],
                                                            mask_zero=output_embedding["mask_zero"]))
        self.output_layer = tf.keras.layers.Dense(units=output_embedding["input_dim"], bias=True)

    def call(self, inputs):
        encoder_output = self.encoder(inputs[0])
        decoder_input = self.decoder_embedding(inputs[1])
        return self.output_layer(self.decoder((encoder_output, decoder_input)))
