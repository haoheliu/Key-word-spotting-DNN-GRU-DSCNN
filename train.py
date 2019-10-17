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

print("Construct dataLoader...")
dataloader = dataLoader()

model = Model()
print("Construct model...")
if(Config.modelName == "DNN_6_512" or Config.modelName == "DNN_3_128"):
    loss, _ = model.lossFunc(batch_ph, label_ph)
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

if(Config.testMode == True):
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, save_path="./models/GRU/model.ckpt")
        data, labels, length,fnames = dataloader.getGRUTestNextBatch()
        temp = sess.run(output, feed_dict={batch_ph: data, length_ph: length})
        for i, fname in enumerate(fnames):
            dataloader.util.plotFileWave(fname, temp[i,:int(length[i]), :])
else:
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
            if(Config.enableVisualize == True):
                print("Start testing...")
                if(Config.drowROC == True):
                    peakVal = []
                    labels = []
                    while (1):
                        data, label = dataloader.getSingleTestData()
                        if (data.shape[0] == 0):
                            break
                        temp = sess.run(output, feed_dict={batch_ph: data})
                        peakVal.append(np.max(posteriorHandling(temp)))
                        labels.append(label)
                    dataloader.util.savePkl("./pickles/CNNpeakVal.pkl", peakVal)
                    dataloader.util.savePkl("./pickles/CNNlabels.pkl", labels)
                    util = dataloader.util
                    gruxcoord, gruycoord = util.drawROC("./pickles/labels.pkl", "./pickles/peakVal.pkl")
                    cnnxcoord, cnnycoord = util.drawROC("./pickles/CNNlabels.pkl", "./pickles/CNNpeakVal.pkl")
                    xcoord_deep, ycoord_deep = util.drawROC("./pickles/DeepDesiredLabel.pkl", "./pickles/DeepConfidence.pkl")
                    xcoord, ycoord = util.drawROC("./pickles/desiredLabel.pkl", "./pickles/confidence.pkl")
                    plt.figure()
                    plt.xlim((0, 0.4))
                    plt.ylim((0, 0.2))
                    plt.text(0.5, 0.5, "W_max: " + str(Config.w_max))
                    plt.text(0.5, 0.7, "W_smooth: " + str(Config.w_smooth))
                    plt.xlabel("False positive rate")
                    plt.ylabel("False reject rate")
                    plt.title("Comparision of three models")
                    plt.scatter(gruxcoord, gruycoord, s=1, label="GRU_128")
                    plt.scatter(xcoord_deep, ycoord_deep, s=1, label="DNN_512_6")
                    plt.scatter(xcoord, ycoord, s=1, label="DNN_128_3")
                    plt.scatter(cnnxcoord, cnnycoord, s=1, label="DSCNN")
                    plt.legend()
                    plt.savefig("./images/ROC/"+str(Config.numEpochs)+"_"+str(Config.w_smooth)+"_"+str(Config.w_max))
                if(Config.visualizeTestData == True):
                    while (1):
                        data,_ = dataloader.getSingleTestData()
                        if (data.shape[0] == 0):
                            break
                        temp = sess.run(output, feed_dict={batch_ph: data})
                        fname = dataloader.testDataFiles[dataloader.currentTestDataFile-1]
                        dataloader.util.plotFileWave(fname,temp)
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
                    summary, repoLoss, _ = sess.run([merged, loss, trainStep],
                                                    feed_dict={batch_ph: data, label_ph: labels, length_ph: length,
                                                               maxLength_ph: [maxLength]})
                elif(Config.modelName == "DSCNN"):
                    summary,_,lossRes = sess.run([merged,trainStep,loss],feed_dict={batch_ph:data,label_ph:labels,length_ph:data.shape[0]})
                elif(Config.modelName in ["DNN_6_512","DNN_3_128"]):
                    pass
                summary_writer.add_summary(summary, counter)

            if(Config.saveModel == True):
                model.save(sess)


