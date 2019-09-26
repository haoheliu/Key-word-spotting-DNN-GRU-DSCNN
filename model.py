import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense

class Model:
    def __init__(self):
        self.model = None
        self.createModel()
        self.trainable_variables = self.model.trainable_variables

    def createModel(self):
        self.model = tf.keras.Sequential()
        self.model.add(Dense(128, activation=tf.nn.relu, input_shape=(1640,)))
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dense(128, activation=tf.nn.relu))
        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

    def __call__(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    m = Model()
