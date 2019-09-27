from util import Util
from config import Config

import numpy as np
import random

class dataLoader:
    def __init__(self):
        self.util = Util()
        self.currentTestPositiveDataFile, self.currentTestNegativeDataFile = 0,0
        self.currentTrainDataFile = 0

        Config.testBatchSize = int((len(self.util.testPositiveDataFiles)+len(self.util.testNegativeDataFiles)) * 0.1)
        self.trainDataFiles = self.util.trainNegativeDataFiles+self.util.trainPositiveDataFiles
        random.shuffle(self.trainDataFiles)

        self.maxTestCacheSize = Config.testBatchSize * Config.maximumFrameNumbers
        self.maxTrainCacheSize = Config.trainBatchSize * Config.maximumFrameNumbers
        self.trainData = {
            "data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.empty(shape=[self.maxTrainCacheSize,3])
        }
        self.testPositiveData = {
            "data": np.zeros(shape=[self.maxTestCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.zeros(shape=[self.maxTestCacheSize,3])
        }
        self.testNegativeData = {
            "data": np.zeros(shape=[self.maxTestCacheSize, (Config.leftFrames + Config.rightFrames + 1) * 40]),
            "label": np.zeros(shape=[self.maxTestCacheSize, 3])
        }

        self.constructTestPositive()

    # Get a batch of positive training example
    def getTrainNextBatch(self):
        self.resetDataDict()
        counter,currentRow = 0,0
        if(self.currentTrainDataFile % 500 == 0):
            print(str(self.currentTrainDataFile)+" training files finished!")
        for i in range(Config.trainBatchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0 # repeat the hole dataset again
                Config.numEpochs -= 1
                if(Config.shuffle == True):
                    print("Shuffle training data ...")
                    random.shuffle(self.trainDataFiles)
                return np.empty(shape=[0]),np.empty(shape=[0])
            fname = self.util.splitFileName(self.util.trainPositiveDataFiles[self.currentTrainDataFile])
            try:
                result = np.load(Config.offlineDataPath+fname+"_data.npy")
                label = np.load(Config.offlineDataPath+fname+"_label.npy")
            except:
                print("Error while reading file: "+fname)
                continue
            self.currentTrainDataFile += 1
            for data,label in zip(result,label):
                self.trainData['data'][currentRow] = data
                self.trainData['label'][currentRow] = label
                currentRow += 1
        self.trainData['data'] = self.trainData['data'][:currentRow]
        self.trainData['label'] = self.trainData['label'][:currentRow]
        return self.trainData['data'],self.trainData['label']

    # Get a batch of positive test example
    def constructTestPositive(self):
        self.resetDataDict()
        self.currentTestDataFile = 0  # repeat the hole dataset again
        currentRow,counter = 0,0
        for i in range(Config.testBatchSize): # All the test files
            if(counter % 100 == 0):
                print("Loading "+str(counter)+" testfiles finished!")
            counter += 1
            fname = self.util.splitFileName(self.util.testPositiveDataFiles[self.currentTestDataFile])
            result = np.load("./offlineDataPositive/"+fname+"_data.npy")
            label = np.load("./offlineDataPositive/"+fname+"_label.npy")
            self.currentTestDataFile += 1
            for data,label in zip(result,label):
                self.testData['data'][currentRow] = data
                self.testData['label'][currentRow] = label
                currentRow += 1
        self.testData['data'] = self.testData['data'][:currentRow]
        self.testData['label'] = self.testData['label'][:currentRow]

    def constructTestNegative(self):
        pass

    def getTestNegative(self):
        pass

    def getTestPositive(self):
        return self.testData['data'],self.testData['label']

    def resetDataDict(self):
        self.trainData = {"data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]), "label": np.empty(shape=[self.maxTrainCacheSize,3])}

if __name__ == "__main__":
    dataloader = dataLoader(dataFileBatchSize=5)
    data,label = dataloader.getTestPositive()
    print(data.shape)
    print(label.shape)