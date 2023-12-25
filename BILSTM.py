import tensorflow as tf
from keras.layers import Input, LSTM, Dropout, Dense, Flatten, Bidirectional, Embedding
from keras.models import Model


# this is the implementation of the model which used in this paper:
# "A bidirectional LSTM deep learning approach for intrusion detection"
# length_in is equal to length dataframe which is going to be trained by this model
class BiLstm(Model):
    def __init__(self, in_shape: tuple = (41,), out_shape: int = 2, length_in=None):
        super(BiLstm, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        model_input = Input(shape=self.in_shape)

        # Embedding Layer
        x = Embedding(input_dim=length_in, input_length=in_shape[0], output_dim=64)(model_input)

        # Bidirectional LSTM layers
        forward_layer = LSTM(64, return_sequences=True)
        backward_layer = LSTM(64, return_sequences=True, go_backwards=True)
        lstm_layer = Bidirectional(forward_layer,
                                   backward_layer=backward_layer,
                                   merge_mode='concat')(x)
        x = Dropout(0.2)(lstm_layer)

        forward_layer2 = LSTM(32, return_sequences=True)
        backward_layer2 = LSTM(32, return_sequences=True, go_backwards=True)
        lstm_layer2 = Bidirectional(forward_layer2,
                                    backward_layer=backward_layer2,
                                    merge_mode='concat')(x)
        x = Dropout(0.2)(lstm_layer2)

        x = Flatten()(x)

        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)

        # Output layer for classification
        output = Dense(self.out_shape,
                       activation='softmax')(x)

        self.model = tf.keras.Model(inputs=model_input, outputs=output)

        self.initialize()
        # self.model.summary()

    def call(self, inputs):
        return self.model(inputs)

    def initialize(self):
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal()
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def build_graph(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    cf = BiLstm((41,), length_in=125973)
    cf.build_graph()
