import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warnings
import tensorflow as tf


def create_model(input_neurons, hidden_layer_neurons=10, hidden_activation='relu', loss='mean_squared_error'):
    inputs = tf.keras.Input(shape=(input_neurons,))
    x = tf.keras.layers.Dense(hidden_layer_neurons, activation=hidden_activation)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer='adam', metrics=['mse'])
    return model
