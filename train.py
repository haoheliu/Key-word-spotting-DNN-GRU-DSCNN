import tensorflow as tf
import os
import numpy as np

from sklearn.utils import shuffle
from model import Model
from dataLoader import dataLoader
from config import Config

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto(allow_soft_placement=True) #log_device_placement=True
config.gpu_options.allow_growth = True
totalEpoches = Config.numEpochs

with tf.name_scope("Placeholder"):
    batch_ph = tf.placeholder(tf.float32,shape=(Config.maximumFrameNumbers,Config.trainBatchSize,40 * (Config.leftFrames + Config.rightFrames + 1)),name='batch_ph')
    label_ph = tf.placeholder(tf.int32, shape=(Config.maximumFrameNumbers,Config.trainBatchSize ), name="label_ph")
    length_ph = tf.placeholder(tf.float32, shape=(Config.trainBatchSize), name="length_ph")

print("Construct dataLoader...")
dataloader = dataLoader()
model = Model()
print("Construct model...")
if(Config.modelName != "GRU"):
    Loss, _ = model.lossFunc(batch_ph, label_ph)
else:
    batchList = []
    targetList = []
    maskList = []
    mask = tf.cast(tf.sequence_mask(lengths=length_ph, maxlen=Config.maximumFrameNumbers),dtype=tf.float32)
    mask = tf.transpose(mask)
    output = model.GRU(batch_ph,length_ph)
    tf.add_to_collection("Pred_network",output)

    for i in range(Config.maximumFrameNumbers):
        batchList.append(output[i])
        targetList.append(label_ph[i])
        maskList.append(mask[i])
    loss = model.sequence_loss(batchList,targetList,maskList)
    tf.add_to_collection("Pred_network", loss)

print("Model Ready!")
print("Construct optimizer...")
with tf.name_scope("modelOptimizer"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(Config.learningRate,
                                               global_step=global_step,
                                               decay_steps=int(10000/Config.trainBatchSize),
                                               decay_rate=Config.decay_rate,
                                               staircase=False)
    trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate,name="gradient_optimizer").minimize(loss,global_step=global_step)
print("Optimizer Ready!")

if(Config.testMode == True):
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, save_path="./models/GRU/model.ckpt")
        data, labels, length,fnames = dataloader.getGRUTestNextBatch()
        temp = sess.run(output, feed_dict={batch_ph: data, length_ph: length})
        for i, fname in enumerate(fnames):
            dataloader.util.plotFileWave(fname, temp[:int(length[i]), i, :])
else:
    print("Start Training Session...")
    with tf.Session(config=config) as sess:
        print("Initialize variables...")
        # Save graph
        if(Config.useTensorBoard == True):
            if(not os.path.exists("./log")):
                os.mkdir("./log")
            writer = tf.summary.FileWriter("./log", tf.get_default_graph())
            writer.close()
        sess.run(tf.global_variables_initializer())
        # Start training
        while(not Config.numEpochs == 0):
            # print("Start testing...")
            # data, label, length, fnames = dataloader.getGRUTestNextBatch()
            # if(data.shape[0] == 0):
            #     data, label, length, fnames = dataloader.getGRUTestNextBatch()
            # temp = sess.run(output, feed_dict={batch_ph: data, length_ph: length})
            # for i, fname in enumerate(fnames):
            #     dataloader.util.plotFileWave(fname, temp[:int(length[i]), i, :])
            currentEpoch = Config.numEpochs
            print("Saving session!")
            model.save(sess)
            print("[EPOCH " + str(totalEpoches - Config.numEpochs + 1), "]", "lr: ", sess.run(learning_rate))
            while(1):
                if(Config.modelName != "GRU"):
                    batchTrain,labelTrain = dataloader.getTrainNextBatch() # Get a batch of data
                else:
                    data, labels, length = dataloader.getGRUTrainNextBatch()
                if(not currentEpoch == Config.numEpochs):
                    break
                if(Config.modelName != "GRU"):
                    pass
                else:
                    repoLoss,_ = sess.run([loss,trainStep],feed_dict={batch_ph:data,label_ph:labels,length_ph:length})
                print(repoLoss)


