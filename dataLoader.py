from util import Util
from config import Config
from PosteriorHandling import posteriorHandling

import numpy as np
import random
import tensorflow as tf
from model import Model
import os

class dataLoader:
    def __init__(self):
        self.util = Util()
        self.currentTestDataFile = 0
        self.currentTrainDataFile = 0

        self.trainDataFiles = self.util.trainPositiveDataFiles +self.util.trainNegativeDataFiles
        self.testDataFiles = self.util.testPositiveDataFiles +self.util.testNegativeDataFiles
        # self.testDataFiles = ["positive_00001.fbank", "positive_00002.fbank", "positive_00003.fbank",
        #                       "positive_00004.fbank", "positive_00005.fbank", "positive_00006.fbank",
        #                       "positive_00009.fbank", "positive_00008.fbank", "positive_00007.fbank",
        #                       "negative_00001.fbank", "negative_00002.fbank", "negative_00003.fbank",
        #                       "negative_00004.fbank", "negative_00005.fbank", "negative_00006.fbank",
        #                       "negative_00009.fbank", "negative_00008.fbank", "negative_00007.fbank"
        #                       ]
        # self.testDataFiles = ["positive_00001.fbank",
        #                       "negative_00001.fbank"
        #                       ]
        self.maxTestCacheSize = Config.testBatchSize * Config.maximumFrameNumbers
        self.maxTrainCacheSize = Config.trainBatchSize * Config.maximumFrameNumbers
        self.trainData = {
            "data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.empty(shape=[self.maxTrainCacheSize,3])
        }
        self.testData = {
            "data": [],
            "label": []
        }

    # Get a batch of positive training example for gru
    def getGRUTestNextBatch(self):
        # Reset
        self.testData = {
            "data": np.zeros(shape=[Config.maximumFrameNumbers,Config.testBatchSize, (Config.leftFrames + Config.rightFrames + 1) * 40]),
            "label": np.zeros(shape=[Config.maximumFrameNumbers,Config.testBatchSize]),
            "length":np.zeros(shape=[Config.testBatchSize]),
            "fnames":[]
        }
        # Report
        if (self.currentTestDataFile % 1000 == 0):
            print(str(self.currentTestDataFile) + " test files finished!")
        for i in range(Config.testBatchSize):
            if (self.currentTestDataFile >= len(self.testDataFiles)):
                self.currentTestDataFile = 0  # repeat the hole dataset again
                if (Config.shuffle == True):
                    print("Shuffle test data ...")
                    random.shuffle(self.testDataFiles)
                return np.empty(shape=[0]), np.empty(shape=[0]),np.empty(shape=[0]),np.empty(shape=[0])
            fname = self.util.splitFileName(self.testDataFiles[self.currentTestDataFile])
            self.testData["fnames"].append(fname)
            try:
                result = np.load(Config.offlineDataPath + fname + "_data.npy")
                label = np.load(Config.offlineDataPath + fname + "_label.npy")
            except Exception as e:
                print("Error while reading file: " + fname,e)
                self.currentTestDataFile += 1
                continue
            self.currentTestDataFile += 1
            currentRow= 0
            for data, label in zip(result, label):
                self.testData['data'][currentRow][i] = data
                self.testData['label'][currentRow][i] = np.argmax(label)
                currentRow += 1
            self.testData['length'][i] = currentRow
        return self.testData['data'], self.testData['label'],self.testData['length'],self.testData["fnames"]

    # Get a batch of positive training example for gru
    def getGRUTrainNextBatch(self):
        # Reset
        self.trainData = {
            "data": np.zeros(shape=[Config.maximumFrameNumbers,Config.trainBatchSize, (Config.leftFrames + Config.rightFrames + 1) * 40]),
            "label": np.zeros(shape=[Config.maximumFrameNumbers,Config.trainBatchSize]),
            "length":np.zeros(shape=[Config.trainBatchSize])
        }
        counter = 0
        # Report
        if (self.currentTrainDataFile % 1000 == 0):
            print(str(self.currentTrainDataFile) + " training files finished!")
        for i in range(Config.trainBatchSize):
            if (self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0  # repeat the hole dataset again
                Config.numEpochs -= 1
                if (Config.shuffle == True):
                    print("Shuffle training data ...")
                    random.shuffle(self.trainDataFiles)
                return np.empty(shape=[0]), np.empty(shape=[0]),np.empty(shape=[0])
            fname = self.util.splitFileName(self.trainDataFiles[self.currentTrainDataFile])
            try:
                result = np.load(Config.offlineDataPath + fname + "_data.npy")
                label = np.load(Config.offlineDataPath + fname + "_label.npy")
            except Exception as e:
                print("Error while reading file: " + fname,e)
                self.currentTrainDataFile += 1
                continue
            self.currentTrainDataFile += 1
            currentRow= 0
            for data, label in zip(result, label):
                self.trainData['data'][currentRow][i] = data
                self.trainData['label'][currentRow][i] = np.argmax(label)
                currentRow += 1
            self.trainData['length'][i] = currentRow
        return self.trainData['data'], self.trainData['label'],self.trainData['length']

    # Get a batch of positive training example
    def getTrainNextBatch(self):
        # Reset
        self.trainData = {
            "data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.empty(shape=[self.maxTrainCacheSize,3]),
        }
        counter,currentRow = 0,0
        # Report
        if(self.currentTrainDataFile % 1000 == 0):
            print(str(self.currentTrainDataFile)+" training files finished!")
        for i in range(Config.trainBatchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0 # repeat the hole dataset again
                Config.numEpochs -= 1
                if(Config.shuffle == True):
                    print("Shuffle training data ...")
                    random.shuffle(self.trainDataFiles)
                return np.empty(shape=[0]),np.empty(shape=[0])
            fname = self.util.splitFileName(self.trainDataFiles[self.currentTrainDataFile])
            try:
                result = np.load(Config.offlineDataPath+fname+"_data.npy")
                label = np.load(Config.offlineDataPath+fname+"_label.npy")
            except:
                print("Error while reading file: "+fname)
                self.currentTrainDataFile += 1
                continue
            self.currentTrainDataFile += 1
            for data,label in zip(result,label):
                self.trainData['data'][currentRow] = data
                self.trainData['label'][currentRow] = label
                currentRow += 1
        self.trainData['data'] = self.trainData['data'][:currentRow]
        self.trainData['label'] = self.trainData['label'][:currentRow]
        return self.trainData['data'],self.trainData['label']

    def getSingleTestData(self,fPath = None):
        if(self.currentTestDataFile >= len(self.testDataFiles)-0):
            self.currentTestDataFile = 0
            random.shuffle(self.testDataFiles)
            return [],[]
        if(not fPath == None):
            fname = self.util.splitFileName(fPath)
        else:
            fname = self.util.splitFileName(self.testDataFiles[self.currentTestDataFile])
        try:
            result = np.load(Config.offlineDataPath + fname + "_data.npy")
            label = np.load(Config.offlineDataPath + fname + "_label.npy")
        except:
            print("Error while reading file: " + fname)
            return [],[]
        testData = {
            "data": np.zeros(shape=[result.shape[0], (Config.leftFrames + Config.rightFrames + 1) * 40]),
            "label": None
        }
        currentRow = 0
        self.currentTestDataFile += 1
        for data, label in zip(result, label):
            testData['data'][currentRow] = data
            currentRow += 1
        type = fname.strip().split('_')[0]
        if(type == 'positive'):
            testData['label'] = 1
        elif(type == 'negative'):
            testData['label'] = 0
        else:
            raise ValueError("File should either be positive or negative")
        return testData['data'],testData['label']

    def constructTestDataSet(self):
        counter = 0
        while(self.currentTestDataFile != 0):
            if(counter % 500 == 1):
                print(str(counter)," test filed loaded,","total test files: "+str(len(self.testDataFiles)))
            data,label = self.getSingleTestData()
            self.testData['data'].append(data)
            self.testData['label'].append(label)
            counter += 1

    def getTestData(self):
        return self.testData['data'],self.testData['label']

    def visualizeROC(self,fileNames,sess,model):
        confidence = []
        desiredLable = []
        for count,file in enumerate(fileNames):
            if(count % 10 == 0):
                print(count)
            try:
                testData, testLabel = self.getSingleTestData(fPath=file)
            except:
                print("Error:" + file)
                continue
            try:
                if(Config.modelName == "GRU"):
                    testData = np.reshape(testData,newshape=(testData.shape[0],1,testData.shape[1]))
                modelOutput = sess.run(model.testModel(),feed_dict={model.TestInput:testData})
            except:
                print("Exception:",file,testData)
            # # modelOutput += Config.base
            confidence.append(np.max(posteriorHandling(modelOutput)))
            self.util.plotFileWave(file,modelOutput=modelOutput)
            if('positive' in file):
                desiredLable .append(1)
            else:
                desiredLable.append(0)
        auc = self.util.plotRoc(desiredLable,confidence)
        if(not os.path.exists("./pickles")):
            os.mkdir("./pickles")
        if(not os.path.exists("./pickles/"+Config.modelName+"/")):
            os.mkdir("./pickles/"+Config.modelName+"/")
        self.util.savePkl("pickles/"+Config.modelName+"/DesiredLabel.pkl",desiredLable)
        self.util.savePkl("pickles/"+Config.modelName+"/Confidence.pkl",confidence)
        return auc

    def visualizaPositiveDataFiles(self,fileNames,sess,model):
        for file in fileNames:
            try:
                testData, testLabel = self.getSingleTestData(fPath=file)
                modelOutput = model(tf.convert_to_tensor(testData, dtype=tf.float32))
            except:
                print("Error:" + file)
                continue
            modelOutput = sess.run(modelOutput)
            # modelOutput += Config.base

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
    config.gpu_options.allow_growth = True

    dataloader = dataLoader()
    # model = Model()
    data,label,length,fnames = dataloader.getGRUTestNextBatch()
    print(data.shape,label.shape,length,length[0],fnames)
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
    #     saver = tf.train.import_meta_graph('./models/GRU/model.ckpt.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint("./models/GRU"))
    #     dataloader.visualizeROC(dataloader.testDataFiles,sess,model)

