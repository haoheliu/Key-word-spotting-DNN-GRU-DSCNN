from util import Util
from config import Config

import numpy as np
import random
import tensorflow as tf

class dataLoader:
    def __init__(self):
        self.util = Util()
        self.currentTestDataFile = 1
        self.currentTrainDataFile = 0

        self.trainDataFiles = self.util.trainPositiveDataFiles +self.util.trainNegativeDataFiles
        self.testDataFiles = ["positive_00001.fbank","positive_00002.fbank","positive_00003.fbank",
                              "positive_00004.fbank","positive_00005.fbank","positive_00006.fbank",
                              "positive_00009.fbank","positive_00008.fbank","positive_00007.fbank",
                              "negative_00001.fbank", "negative_00002.fbank", "negative_00003.fbank",
                              "negative_00004.fbank", "negative_00005.fbank", "negative_00006.fbank",
                              "negative_00009.fbank", "negative_00008.fbank", "negative_00007.fbank"
                              ]
        # self.util.testPositiveDataFiles[:10]+self.util.testNegativeDataFiles[:50]
        # random.shuffle(self.trainDataFiles)
        # random.shuffle(self.testDataFiles)
        # self.testDataFiles = self.testDataFiles[:100]
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
        # self.constructTestDataSet()

    # Get a batch of positive training example
    def getTrainNextBatch(self):
        # Reset
        self.trainData = {
            "data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.empty(shape=[self.maxTrainCacheSize,3])
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

    def visualizaPositiveDataFiles(self,fileNames,sess,model):
        for file in fileNames:
            try:
                testData, testLabel = self.getSingleTestData(fPath=file)
                modelOutput = model(tf.convert_to_tensor(testData, dtype=tf.float32))
            except:
                print("Error:" + file)
                continue
            modelOutput = sess.run(modelOutput)
            modelOutput += Config.base
            self.util.plotFileWave(file,modelOutput = modelOutput)

if __name__ == "__main__":
    dataloader = dataLoader()
    data,label = dataloader.getTrainNextBatch()
    print(label)