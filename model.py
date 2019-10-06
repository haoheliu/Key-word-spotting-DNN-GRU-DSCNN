import tensorflow as tf

from tensorflow.keras.layers import Activation, Dense
from config import Config

class Model:
    def __init__(self):
        self.model = None
        if(Config.useDeepModel == True):
            self.createModel_6_512()
        else:
            self.createModel_3_128()
        self.trainable_variables = self.model.trainable_variables
        self.TestInput = tf.placeholder(tf.float32, shape=(None, 40 * (Config.leftFrames + Config.rightFrames + 1)), name='batch_input')

    def createModel_3_128(self):
        with tf.name_scope("DNNModel_3_128"):
            self.model = tf.keras.Sequential()
            self.model.add(Dense(128, activation=tf.nn.relu, input_shape=(1640,),name="First_layer"))
            self.model.add(Dense(128, activation=tf.nn.relu,name="Second_layer"))
            self.model.add(Dense(128, activation=tf.nn.relu,name="Third_layer"))
            self.model.add(Dense(3,name="Output_layer"))
            # self.model.add(Activation('softmax',name="SOFTMAX_activation"))

    def createModel_6_512(self):
        with tf.name_scope("DNNModel_6_512"):
            self.model = tf.keras.Sequential()
            self.model.add(Dense(512, activation=tf.nn.relu, input_shape=(1640,)))
            self.model.add(Dense(512, activation=tf.nn.relu))
            self.model.add(Dense(512, activation=tf.nn.relu))
            self.model.add(Dense(512, activation=tf.nn.relu))
            self.model.add(Dense(512, activation=tf.nn.relu))
            self.model.add(Dense(512, activation=tf.nn.relu))
            self.model.add(Dense(3,name="Output_layer"))
            # self.model.add(Activation('softmax',name="SOFTMAX_activation"))

    def __call__(self, inputs):
        return self.model(inputs)

    def testModel(self):
        return self.model(tf.convert_to_tensor(self.TestInput,dtype=tf.float32))

    def lossFunc_Paper(self, inputs, targets):
        with tf.name_scope("lossFunction"):
            modelOutput = self.model(tf.convert_to_tensor(inputs, dtype=tf.float32))
            index = tf.argmax(targets, axis=1,name="argmax_Convert3_1tolabel")
            oneHot = tf.cast(tf.one_hot(index, 3, 1, 0), dtype=tf.float32)
            output = tf.log(tf.reduce_sum(modelOutput * oneHot, 1,name="Add_all"))
            output = -tf.reduce_sum(output)
            return output, modelOutput

    def lossFunc_CrossEntropy(self, inputs, targets):
        modelOutput = self.model(tf.convert_to_tensor(inputs, dtype=tf.float32))
        output = tf.nn.softmax_cross_entropy_with_logits(logits=modelOutput, labels=targets)
        output = tf.reduce_sum(output)
        return output,modelOutput

if __name__ == "__main__":
    a = tf.Variable([[1,0,0],[0,1,0],[0,0,1]],dtype=tf.float32)
    b = tf.Variable([[0, 0, 0], [0, 1, 0], [0, 0, 1]],dtype=tf.float32)
    loss1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=a,logits=a))
    loss2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=b, logits=b))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(loss1))
        print(sess.run(loss2))