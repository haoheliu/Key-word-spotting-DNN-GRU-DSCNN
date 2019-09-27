import tensorflow as tf
import numpy as np
import os

from model import Model
from dataLoader import dataLoader
from config import Config


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
label = tf.placeholder(tf.float32,shape = (None,3),name="label")

# saver = tf.train.Saver()

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
    confidence[0] = (modelOutput[0][1] * modelOutput[0][2]) ** 0.5
    for i in range(2, modelOutput.shape[0] + 1):
        h_smooth = max(1, i - Config.w_smooth + 1)
        modelOutput[i - 1] = modelOutput[h_smooth:i].sum(axis=0) / (i - h_smooth + 1)
        h_max = max(1, i - Config.w_max + 1)
        windowMax = np.max(modelOutput[h_max:i], axis=0)
        confidence[i - 1] = (windowMax[1] * windowMax[2]) ** 0.5
    return np.max(confidence)

Loss,_ = dnnModel.lossFunc(batch, label)

print("Construct optimizer...")
trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate).minimize(Loss)

testBatch, testLabel = dataloader.getTestData()
counter = 0

print("Start Training Session...")
with tf.Session(config=config) as sess:
    print("Initialize variables...")
    sess.run(tf.global_variables_initializer())
    while(not Config.numEpochs == 0):
        currentEpoch = Config.numEpochs
        if(Config.numEpochs % 1 == 0):
            print("Start testing... ", end="")
            confidence = []
            labels = []
            while(1):
                # counter += 1
                # if(counter % 10 == 0):
                #     print(counter)
                testData,testLabel = dataloader.getSingleTestData()
                if(len(testData) == 0 and len(testLabel) == 0):
                    break
                modelOutput = dnnModel.model(tf.convert_to_tensor(testData, dtype=tf.float32))
                modelOutput = sess.run(modelOutput)
                confidence.append(posteriorHandling(modelOutput))
                labels.append(testLabel)

            auc = dataloader.util.plotRoc(labels,confidence)
            print("auc",auc)

        print("[EPOCH " + str(totalEpoches - Config.numEpochs), "]")
        while(1):
            batchTrain,labelTrain = dataloader.getTrainNextBatch() # Get a batch of data
            if(not currentEpoch == Config.numEpochs):
                break
            sess.run(trainStep,feed_dict={batch:batchTrain,label:labelTrain})
        continue
