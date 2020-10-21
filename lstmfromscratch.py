
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LSTM(layers.Layer):
    """
    LSTM implementation
    """
    def __init__(self, input_dim = 4, output_dim = 2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        weight_init = tf.random_normal_initializer()
        bias_init = tf.zeros_initializer()
        self.Ui = tf.Variable(initial_value=weight_init(shape=(self.input_dim, self.output_dim), dtype="float32"), trainable=True)
        self.Wi = tf.Variable(initial_value=weight_init(shape=(self.output_dim, self.output_dim), dtype="float32"), trainable=True)
        self.bi = tf.Variable(initial_value=bias_init(shape=(self.output_dim,), dtype="float32"), trainable=True)
        self.Uf = tf.Variable(initial_value=weight_init(shape=(self.input_dim, self.output_dim), dtype="float32"), trainable=True)
        self.Wf = tf.Variable(initial_value=weight_init(shape=(self.output_dim, self.output_dim), dtype="float32"), trainable=True)
        self.bf = tf.Variable(initial_value=bias_init(shape=(self.output_dim,), dtype="float32"), trainable=True)
        self.Uo = tf.Variable(initial_value=weight_init(shape=(self.input_dim, self.output_dim), dtype="float32"), trainable=True)
        self.Wo = tf.Variable(initial_value=weight_init(shape=(self.output_dim, self.output_dim), dtype="float32"), trainable=True)
        self.bo = tf.Variable(initial_value=bias_init(shape=(self.output_dim,), dtype="float32"), trainable=True)
        self.Ug = tf.Variable(initial_value=weight_init(shape=(self.input_dim, self.output_dim), dtype="float32"), trainable=True)
        self.Wg = tf.Variable(initial_value=weight_init(shape=(self.output_dim, self.output_dim), dtype="float32"), trainable=True)
        self.bg = tf.Variable(initial_value=bias_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

    def callForSingleElement(self, cur_input_vec, prev_h_vec, prev_state_vec):
        i_t = tf.math.sigmoid(tf.matmul(cur_input_vec, self.Ui) + tf.matmul(prev_h_vec, self.Wi) + self.bi)
        f_t = tf.math.sigmoid(tf.matmul(cur_input_vec, self.Uf) + tf.matmul(prev_h_vec, self.Wf) + self.bf)
        o_t = tf.math.sigmoid(tf.matmul(cur_input_vec, self.Uo) + tf.matmul(prev_h_vec, self.Wo) + self.bo)
        c_t_candid = tf.math.tanh(tf.matmul(cur_input_vec, self.Ug) + tf.matmul(prev_h_vec, self.Wg) + self.bg)
        c_t = tf.math.sigmoid(tf.math.multiply(f_t, prev_state_vec) + tf.math.multiply(i_t, c_t_candid))
        h_t = tf.math.multiply(tf.math.tanh(c_t), o_t)
        return [c_t, h_t]

    def call(self, seq_input_vec):
        """
        seq_input_vec is assumed to be of shape (num_batches, num_sequence, input_dim)
        Of course, sequences can be of variable length. Meaningless elements at the end are
        assumed to be represented with the zero vector.
        """
        num_batches = seq_input_vec.shape[0]
        num_sequence = seq_input_vec.shape[1]
        input_dim = self.input_dim

        outputs = []

        cur_h_vec = tf.zeros(shape=(num_batches, self.input_dim))
        cur_state_vec = tf.zeros(shape=(num_batches, self.input_dim))

        seq_counter = 0
        while seq_counter < num_sequence:
            cur_batch_in_seq = seq_input_vec[:, seq_counter, :]
            [cur_state_vec, cur_h_vec] = self.callForSingleElement(
                    cur_batch_in_seq,
                    cur_h_vec,
                    cur_state_vec
            )
            outputs.append(cur_h_vec)
            seq_counter += 1

        return tf.stack(outputs, axis=1)
