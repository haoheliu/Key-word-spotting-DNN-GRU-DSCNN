import tensorflow as tf

from tensorflow.keras.layers import Activation, Dense

class Model:
    def __init__(self):
        self.model = None
        self.createModel()
        self.trainable_variables = self.model.trainable_variables

    def createModel(self):
        with tf.name_scope("DNNModel"):
            self.model = tf.keras.Sequential()
            self.model.add(Dense(128, activation=tf.nn.relu, input_shape=(1640,),name="First_layer"))
            self.model.add(Dense(128, activation=tf.nn.relu,name="Second_layer"))
            self.model.add(Dense(128, activation=tf.nn.relu,name="Third_layer"))
            self.model.add(Dense(3,name="Output_layer"))
            self.model.add(Activation('softmax',name="SOFTMAX_activation"))

    def __call__(self, inputs):
        return self.model(inputs)

    def lossFunc_Paper(self, inputs, targets):
        with tf.name_scope("lossFunction"):
            modelOutput = self.model(tf.convert_to_tensor(inputs, dtype=tf.float32))
            index = tf.argmax(targets, axis=1,name="argmax_Convert3_1tolabel")
            oneHot = tf.one_hot(index, 3, 1, 0)
            oneHot = tf.cast(oneHot, dtype=tf.float32)
            output = tf.reduce_sum(modelOutput * oneHot, 1,name="Add_all")
            output = tf.log(output)
            output = -tf.reduce_sum(output)
            return output, modelOutput

    def lossFunc_CrossEntropy(self, inputs, targets):
        modelOutput = self.model(tf.convert_to_tensor(inputs, dtype=tf.float32))
        output = tf.nn.softmax_cross_entropy_with_logits(logits=modelOutput, labels=targets)
        output = tf.reduce_sum(output)
        return output,modelOutput

if __name__ == "__main__":
    model = Model()
