import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from model import Model
from dataLoader import dataLoader
from config import Config

def calculateAccuracy(output,desired):
    assert (output.shape == desired.shape)
    length = output.shape[0]
    same = 0
    for i,j in zip(output,desired):
        if(i == j):
            same += 1
    return same/length

def posteriorHandling(modelOutput):
    confidence = np.zeros(shape=(modelOutput.shape[0]))
    confidence[0] = (modelOutput[0][1] * modelOutput[0][2])# ** 0.5
    for i in range(2, modelOutput.shape[0] + 1):
        h_smooth = max(1, i - Config.w_smooth + 1)
        modelOutput[i - 1] = modelOutput[h_smooth:i].sum(axis=0) / (i - h_smooth + 1)
        h_max = max(1, i - Config.w_max + 1)
        windowMax = np.max(modelOutput[h_max:i], axis=0)
        confidence[i - 1] = (windowMax[1] * windowMax[2])# ** 0.5
    return np.max(confidence)



os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto(allow_soft_placement=True) #log_device_placement=True
config.gpu_options.allow_growth = True
totalEpoches = Config.numEpochs

print("Construct model...")
dnnModel = Model()
print("Construct dataLoader...")
dataloader = dataLoader()
batch = tf.placeholder(tf.float32,shape = (None,40*(Config.leftFrames+Config.rightFrames+1)),name = 'batch_input')
label = tf.placeholder(tf.float32,shape = (None,3),name="label_input")
Loss,_ = dnnModel.lossFunc_CrossEntropy(batch, label)
# saver = tf.train.Saver()
print("Construct optimizer...")
with tf.name_scope("modelOptimizer"):
    trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate,name="gradient_optimizer").minimize(Loss)

testData,testLabel = dataloader.getSingleTestData(fPath="positive_00001.fbank")

print("Start Training Session...")
with tf.Session(config=config) as sess:
    print("Initialize variables...")
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    # writer.close()
    # exit(0)
    while(not Config.numEpochs == 0):
        currentEpoch = Config.numEpochs
        if(Config.numEpochs % 1 == 0):
            print("Start testing... ", end="")
            confidence = []
            labels = []
            while(1):
                if(len(testData) == 0 and len(testLabel) == 0):
                    break
                modelOutput = dnnModel.model(tf.convert_to_tensor(testData, dtype=tf.float32))
                modelOutput = sess.run(modelOutput)

                visual = modelOutput.T
                plt.subplot(211)
                plt.plot(visual[0], color='r', label="filler")
                plt.legend()
                plt.xlabel("frames")
                plt.ylabel("confidence")
                plt.subplot(212)
                plt.plot(visual[1], color='b', label="hello")
                plt.plot(visual[2], color='g', label="xiaogua")
                plt.legend()
                plt.xlabel("frames")
                plt.ylabel("confidence")
                plt.show()
                break
            # auc = dataloader.util.plotRoc(labels,confidence)
            # print("auc",auc)

        print("[EPOCH " + str(totalEpoches - Config.numEpochs), "]")
        while(1):
            batchTrain,labelTrain = dataloader.getTrainNextBatch() # Get a batch of data
            batchTrain, labelTrain = shuffle(batchTrain,labelTrain)
            if(not currentEpoch == Config.numEpochs):
                break
            sess.run(trainStep,feed_dict={batch:batchTrain,label:labelTrain})
        continue

