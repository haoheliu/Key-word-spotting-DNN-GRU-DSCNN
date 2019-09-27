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

    def lossFunc(self, inputs, targets):
        modelOutput = self.model(tf.convert_to_tensor(inputs, dtype=tf.float32))
        index = tf.argmax(targets, axis=1)
        oneHot = tf.one_hot(index, 3, 1, 0)
        oneHot = tf.cast(oneHot, dtype=tf.float32)
        output = tf.reduce_sum(modelOutput * oneHot, 1)
        out = -tf.reduce_sum(output)
        return out, modelOutput

if __name__ == "__main__":
    m = Model()
