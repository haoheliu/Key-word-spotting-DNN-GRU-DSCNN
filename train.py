import tensorflow as tf
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys

from sklearn.utils import shuffle
from model import Model
from dataLoader import dataLoader
from config import Config
from tensorflow.contrib.seq2seq import sequence_loss
from PosteriorHandling import posteriorHandling
import tensorflow.contrib.slim as slim

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto(allow_soft_placement=True) #log_device_placement=True
config.gpu_options.allow_growth = True
totalEpoches = Config.numEpochs

with tf.name_scope("Placeholder"):
    if(Config.modelName == "GRU"):
        batch_ph = tf.placeholder(tf.float32,shape=(Config.trainBatchSize,None,40 * (Config.leftFrames + Config.rightFrames + 1)),name='batch_ph')
        label_ph = tf.placeholder(tf.int32, shape=(Config.trainBatchSize,None), name="label_ph")
        length_ph = tf.placeholder(tf.float32, shape=(Config.trainBatchSize), name="length_ph")
        maxLength_ph = tf.placeholder(tf.int32)
    elif(Config.modelName == "DSCNN"):
        batch_ph = tf.placeholder(tf.float32, [None, Config.featureLength], name='batch_ph')
        label_ph = tf.placeholder(tf.int32, shape=(None, 3), name="label_ph")
        length_ph = tf.placeholder(tf.float32)
    else:
        batch_ph = tf.placeholder(tf.float32, shape=(None, 40 * (Config.leftFrames + Config.rightFrames + 1)),name='batch_input')
        label_ph = tf.placeholder(tf.float32, shape=(None, 3), name="label_input")

print("Construct dataLoader...")
dataloader = dataLoader()

model = Model()
print("Construct model...")
if(Config.modelName == "DNN_6_512" or Config.modelName == "DNN_3_128"):
    loss, _ = model.lossFunc_CrossEntropy(batch_ph, label_ph)
elif(Config.modelName == "DSCNN"):
    output = model.create_ds_cnn_model(batch_ph,Config.model_settings,Config.model_size_info,Config.is_training)
    loss = model.lossFunc_dscnn(output,label_ph)
elif(Config.modelName == "GRU"):
    mask = tf.cast(tf.sequence_mask(lengths=length_ph, maxlen=maxLength_ph[0]),dtype=tf.float32)
    output = model.GRU(batch_ph,length_ph)
    tf.add_to_collection("Pred_network",output)
    loss = sequence_loss(output,label_ph,mask)

tf.summary.scalar("Loss", loss/length_ph)
tf.add_to_collection("Pred_network", output)
tf.add_to_collection("Pred_network", loss)

print("Model Ready!")
print("Construct optimizer...")
with tf.name_scope("modelOptimizer"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(Config.learningRate,
                                               global_step=global_step,
                                               decay_steps=int(10000/Config.trainBatchSize),
                                               decay_rate=Config.decay_rate,
                                               staircase=True)
    tf.summary.scalar("learningRate", learning_rate)
    trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate,name="gradient_optimizer").minimize(loss,global_step=global_step)

print("Optimizer Ready!")

merged = tf.summary.merge_all()
counter =0

print("Start Training Session...")
with tf.Session(config=config) as sess:
    print("Initialize variables...")
    # Save graph
    if(Config.useTensorBoard == True):
        if(not os.path.exists(Config.logDir)):
            os.mkdir(Config.logDir)
        summary_writer = tf.summary.FileWriter(Config.logDir,sess.graph)
    sess.run(tf.global_variables_initializer())
    # Start training
    while(not Config.numEpochs == 0):
        currentEpoch = Config.numEpochs
        print("Saving session!")
        print("[EPOCH " + str(totalEpoches - Config.numEpochs + 1), "]", "lr: ", sess.run(learning_rate))
        while(1):
            counter += 1
            if(Config.modelName in ["DSCNN","DNN_6_512","DNN_3_128"]):
                data,labels = dataloader.getTrainNextBatch() # Get a batch of data
            else:
                data, labels, length = dataloader.getGRUTrainNextBatch()

            if(not currentEpoch == Config.numEpochs or data.shape[0] == 0):
                break

            if(Config.modelName == "GRU"):
                maxLength = np.max(length)
                summary, repoLoss, _ = sess.run([merged, loss, trainStep],feed_dict={batch_ph: data, label_ph: labels, length_ph: length,maxLength_ph: [maxLength]})
            elif(Config.modelName == "DSCNN"):
                summary,_,lossRes = sess.run([merged,trainStep,loss],feed_dict={batch_ph:data,label_ph:labels,length_ph:data.shape[0]})
            elif(Config.modelName in ["DNN_6_512","DNN_3_128"]):
                data, labels = shuffle(data, labels)
                sess.run(trainStep,feed_dict={batch_ph:data,label_ph:labels})

        if (Config.useTensorBoard == True):
            summary_writer.add_summary(summary, counter)

        if(Config.saveModel == True):
            model.save(sess)


