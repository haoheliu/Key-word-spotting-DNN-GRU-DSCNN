import tensorflow as tf
import os

from tensorflow.keras.layers import Activation, Dense, GRU
from config import Config

class Model:
    def __init__(self):
        self.model = None
        if(Config.modelName == "DNN_6_512"):
            self.createModel_6_512()
        elif(Config.modelName == "DNN_3_128"):
            self.createModel_3_128()
        elif(Config.modelName == "GRU"):
            pass

        if(Config.modelName == "GRU"):
            self.TestInput = tf.placeholder(tf.float32, shape=(None,1, 40 * (Config.leftFrames + Config.rightFrames + 1)), name='batch_input')
        else:
            self.TestInput = tf.placeholder(tf.float32, shape=(None, 40 * (Config.leftFrames + Config.rightFrames + 1)), name='batch_input')

        if(Config.lossFunc == "seqLoss"):
            self.lossFunc = self.lossFunc_SequenceLoss
        elif(Config.lossFunc == "crossEntropy"):
            self.lossFunc = self.lossFunc_CrossEntropy
        elif(Config.lossFunc == "Paper"):
            self.lossFunc = self.lossFunc_Paper

    def sequence_loss(self,
                      logits,
                      targets,
                      weights,
                      average_across_timesteps=True,
                      average_across_batch=True,
                      softmax_loss_function=None,
                      name=None):
        with tf.name_scope(name, "sequence_loss", logits + targets + weights):
            cost = tf.reduce_sum(
                self.sequence_loss_by_example(
                    logits,
                    targets,
                    weights,
                    average_across_timesteps=average_across_timesteps,
                    softmax_loss_function=softmax_loss_function))
            if average_across_batch:
                batch_size = tf.shape(targets[0])[0]
                return cost / tf.cast(batch_size, cost.dtype)
            else:
                return cost

    def sequence_loss_by_example(self,
                                 logits,
                                 targets,
                                 weights,
                                 average_across_timesteps=True,
                                 softmax_loss_function=None,
                                 name=None):
        """Weighted cross-entropy loss for a sequence of logits (per example).
        Args:
          logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
          targets: List of 1D batch-sized int32 Tensors of the same length as logits.
          weights: List of 1D batch-sized float-Tensors of the same length as logits.
          average_across_timesteps: If set, divide the returned cost by the total
            label weight.
          softmax_loss_function: Function (labels, logits) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
            **Note that to avoid confusion, it is required for the function to accept
            named arguments.**
          name: Optional name for this operation, default: "sequence_loss_by_example".

        Returns:
          1D batch-sized float Tensor: The log-perplexity for each sequence.

        Raises:
          ValueError: If len(logits) is different from len(targets) or len(weights).
        """
        # 此三者都是列表，长度都应该相同
        if len(targets) != len(logits) or len(weights) != len(logits):
            raise ValueError("Lengths of logits, weights, and targets must be the same "
                             "%d, %d, %d." % (len(logits), len(weights), len(targets)))
        with tf.name_scope(name, "sequence_loss_by_example",
                           logits + targets + weights):
            log_perp_list = []
            # 计算每个时间片的损失
            for logit, target, weight in zip(logits, targets, weights):
                if softmax_loss_function is None:
                    # 默认使用sparse
                    target = tf.reshape(target, [-1])
                    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=target, logits=logit)
                else:
                    crossent = softmax_loss_function(labels=target, logits=logit)
                log_perp_list.append(crossent * weight)
            # 把各个时间片的损失加起来
            log_perps = tf.add_n(log_perp_list)
            # 对各个时间片的损失求平均数
            if average_across_timesteps:
                total_size = tf.add_n(weights)
                total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
                log_perps /= total_size
        return log_perps

    def createModel_3_128(self):
        with tf.name_scope("DNNModel_3_128"):
            self.model = tf.keras.Sequential()
            self.model.add(Dense(128, activation=tf.nn.relu, input_shape=(1640,),name="First_layer"))
            self.model.add(Dense(128, activation=tf.nn.relu,name="Second_layer"))
            self.model.add(Dense(128, activation=tf.nn.relu,name="Third_layer"))
            self.model.add(Dense(3,name="Output_layer"))

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

    def GRU(self,batch,length):
        with tf.name_scope("GRU"):
            cell = tf.contrib.rnn.GRUCell(num_units=128)
            outputs, _ = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=length,
                inputs=batch,
                time_major= True)
            outputs = tf.layers.dense(outputs, 3)
        return outputs

    def __call__(self, inputs,mask = None):
        return self.model(inputs = inputs,mask = mask)

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

    #  (1300, 8, 840) (1300, 8) (8,)
    def lossFunc_SequenceLoss(self,inputs,targets,length):
        weight = tf.sequence_mask(length, Config.maximumFrameNumbers)
        output = self.model(inputs = tf.convert_to_tensor(inputs, dtype=tf.float32))
        weight = tf.cast(tf.transpose(weight),dtype = tf.float32)
        targets = tf.cast(targets,dtype=tf.int32)
        # return self.sequence_loss(logits = [inputs], targets = [targets],weights = [weight]),None
        return [output],[targets],[weight]

    def save(self,sess):
        self.saver = tf.train.Saver()
        if(not os.path.exists("./models")):
            os.mkdir("./models")
            os.mkdir("./models/DNN_6_512")
            os.mkdir("./models/DNN_3_128")
            os.mkdir("./models/GRU")
        if(Config.modelName == "DNN_6_512"):
            self.saver.save(sess,"./models/DNN_6_512/model.ckpt")
        elif(Config.modelName == "DNN_3_128"):
            self.saver.save(sess, "./models/DNN_3_128/model.ckpt")
        elif(Config.modelName == "GRU"):
            self.saver.save(sess, "./models/GRU/model.ckpt")

if __name__ == "__main__":
    from dataLoader import dataLoader
    dataloader = dataLoader()

    writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    writer.close()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     ddata,dlabel,dlength = dataloader.getGRUTrainNextBatch() # (1300, 8, 840) (1300, 8) (8,)
    #     output = sess.run([outputs],feed_dict={
    #             batch:ddata,
    #             label:dlabel,
    #             length:dlength
    #     })
    #     print(output)
    #     print(len(output))
    #     print(output[0].shape)


    # input0 = tf.Variable(tf.ones([10,100,1640]))
    # input1 = tf.Variable(tf.ones([100,20,1640]))
    # output0 = model(inputs = input0)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     result0 = sess.run(output0)
    #     print(result0)
    #     print(model.model.summary())