import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

from tensorflow.keras.layers import Activation, Dense, GRU
from config import Config
from PosteriorHandling import posteriorHandling
from util import Util


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
            pass
        else:
            self.TestInput = tf.placeholder(tf.float32, shape=(None, 40 * (Config.leftFrames + Config.rightFrames + 1)), name='batch_input')

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

    def testModel(self):
        return self.model(tf.convert_to_tensor(self.TestInput,dtype=tf.float32))

    def create_ds_cnn_model(self,fingerprint_input, model_settings, model_size_info,
                            is_training):
        """Builds a model with depthwise separable convolutional neural network
        Model definition is based on https://arxiv.org/abs/1704.04861 and
        Tensorflow implementation: https://github.com/Zehaos/MobileNet

        model_size_info: defines number of layers, followed by the DS-Conv layer
          parameters in the order {number of conv features, conv filter height,
          width and stride in y,x dir.} for each of the layers.
        Note that first layer is always regular convolution, but the remaining
          layers are all depthwise separable convolutions.
        """

        def _depthwise_separable_conv(inputs,
                                      num_pwc_filters,
                                      sc,
                                      kernel_size,
                                      stride):
            # skip pointwise by setting num_outputs=None
            depthwise_conv = slim.separable_convolution2d(inputs,
                                                          # A tensor of size [batch_size, height, width, channels].
                                                          num_outputs=None,
                                                          stride=stride,
                                                          depth_multiplier=1,
                                                          kernel_size=kernel_size,
                                                          scope=sc + '/dw_conv')

            bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_conv/batch_norm')
            pointwise_conv = slim.convolution2d(bn,
                                                num_pwc_filters,
                                                kernel_size=[1, 1],
                                                scope=sc + '/pw_conv')
            bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_conv/batch_norm')
            return bn

        label_count = model_settings['label_count']  # 3
        input_frequency_size = model_settings['feature_number']  # 40
        input_time_size = model_settings['stack_frame_number']  # 41
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])
        t_dim = input_time_size
        f_dim = input_frequency_size

        # Extract model dimensions from model_size_info
        num_layers = model_size_info[0]
        conv_feat = [None] * num_layers
        conv_kt = [None] * num_layers  # Conv filter height
        conv_st = [None] * num_layers  # Conv filter width
        conv_kf = [None] * num_layers
        conv_sf = [None] * num_layers
        i = 1
        for layer_no in range(0, num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1

        scope = 'DS-CNN'
        with tf.variable_scope(scope) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                activation_fn=None,
                                weights_initializer=slim.initializers.xavier_initializer(),
                                biases_initializer=slim.init_ops.zeros_initializer(),
                                outputs_collections=[end_points_collection]):
                with slim.arg_scope([slim.batch_norm],
                                    is_training=is_training,
                                    decay=0.96,
                                    updates_collections=None,
                                    activation_fn=tf.nn.relu):
                    for layer_no in range(0, num_layers):
                        if layer_no == 0:
                            net = slim.convolution2d(fingerprint_4d, conv_feat[layer_no], \
                                                     [conv_kt[layer_no], conv_kf[layer_no]],
                                                     stride=[conv_st[layer_no], conv_sf[layer_no]], padding='SAME',
                                                     scope='conv_1')
                            net = tf.cast(net, dtype=tf.float32)
                            net = slim.batch_norm(net, scope='conv_1/batch_norm')
                        else:
                            net = _depthwise_separable_conv(net, conv_feat[layer_no], \
                                                            kernel_size=[conv_kt[layer_no], conv_kf[layer_no]], \
                                                            stride=[conv_st[layer_no], conv_sf[layer_no]],
                                                            sc='conv_ds_' + str(layer_no))
                        t_dim = math.ceil(t_dim / float(conv_st[layer_no]))
                        f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))
                    net = slim.avg_pool2d(net, [t_dim, f_dim], scope='avg_pool')
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            logits = slim.fully_connected(net, label_count, activation_fn=None, scope='fc1')
        return logits

    def GRU(self,batch,length):
        with tf.variable_scope("GRU"):
            cell = tf.contrib.rnn.GRUCell(num_units=128,name = "gru_cell")
            outputs, _ = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=length,
                inputs=batch,
                time_major= False)
            outputs = tf.layers.dense(outputs, 3,name="dense_output")
        return outputs

    def __call__(self, inputs,mask = None):
        return self.model(inputs = inputs,mask = mask)

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

    def lossFunc_dscnn(self, inputs, targets):
        output = tf.nn.softmax_cross_entropy_with_logits(logits=inputs, labels=targets)
        output = tf.reduce_sum(output)
        return output

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
        elif(Config.modelName == "DSCNN"):
            self.saver.save(sess, "./models/DSCNN/model.ckpt")

if __name__ == "__main__":
    model = Model()
    fingerprint_input = tf.placeholder(
        tf.float32, [None, Config.fingerprint_size], name='fingerprint_input')
    logits = model.create_ds_cnn_model(fingerprint_input,Config.model_settings,Config.model_size_info,Config.is_training)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(logits,feed_dict={fingerprint_input:np.random.random((20,840))}))

    exit(0)



    from dataLoader import dataLoader
    dataloader = dataLoader()
    saver = tf.train.import_meta_graph("./models/DSCNN/model.ckpt.meta")
    with tf.Session() as sess:
        saver.restore(sess,"./models/DSCNN/model.ckpt")
        graph = tf.get_default_graph()
        peakVal = []
        labels = []
        predNetwork = graph.get_collection("Pred_network")
        output = predNetwork[0]
        trainInput = graph.get_operation_by_name("Placeholder/batch_ph").outputs[0]
        trainLabel = graph.get_operation_by_name("Placeholder/label_ph").outputs[0]
        while(1):
            data,label,length,fnames = dataloader.getGRUTestNextBatch()
            if(data.shape[0] == 0):
                break
            temp = sess.run(output,feed_dict={trainInput:data,trainLabel:label})
            for i in range(len(fnames)):
                a = int(length[i])
                # dataloader.util.plotFileWave(fnames[i], temp[i,:int(length[i]), :])
                peakVal.append(np.max(posteriorHandling(temp[i,:int(length[i]),:])))
                if("positive" in fnames[i]):
                    labels.append(1)
                else:
                    labels.append(0)
        dataloader.util.savePkl("./pickles/peakVal.pkl",peakVal)
        dataloader.util.savePkl("./pickles/labels.pkl",labels)

    util = Util()
    gruxcoord, gruycoord = util.drawROC("./pickles/labels.pkl", "./pickles/peakVal.pkl")
    xcoord_deep, ycoord_deep = util.drawROC("./pickles/DeepDesiredLabel.pkl", "./pickles/DeepConfidence.pkl")
    xcoord, ycoord = util.drawROC("./pickles/desiredLabel.pkl", "./pickles/confidence.pkl")
    plt.xlim((0,0.4))
    plt.ylim((0,0.2))
    plt.text(0.5, 0.5, "W_max: "+str(Config.w_max))
    plt.text(0.5, 0.7, "W_smooth: "+str(Config.w_smooth))
    plt.xlabel("False positive rate")
    plt.ylabel("False reject rate")
    plt.title("Comparision of three models")
    plt.scatter(gruxcoord, gruycoord, s=1, label="DSCNN")
    plt.scatter(xcoord_deep, ycoord_deep, s=1, label="DNN_512_6")
    plt.scatter(xcoord, ycoord, s=1, label="DNN_128_3")
    plt.legend()
    # plt.savefig(str(Config.w_smooth)+"_"+str(Config.w_max))
    plt.show()


