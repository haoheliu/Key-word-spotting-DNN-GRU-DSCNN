from util import Util
import os
import numpy as np
import tensorflow as tf

class dataLoader:
    def __init__(self,dataFileBatchSize = 2,shuffle = True):
        self.util = Util()
        self.shuffle = shuffle
        self.dataFileBatchSize = dataFileBatchSize
        self.trainData = {"data": [], "label": []}
        self.testData = {"data": [], "label": []}
        self.currentTestDataFile, self.currentTrainDataFile = 0,0
        # self.positiveTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/test/"
        # self.positiveTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/train/"
        self.positiveTestPath = "./data/test/"
        self.positiveTrainPath = "./data/train/"
        self.testDataFiles, self.trainDataFiles = self.constructPositiveDataSet()

    # List All the Positive test&train files
    def constructPositiveDataSet(self):
        testDataFiles, trainDataFiles = os.listdir(self.positiveTestPath), os.listdir(self.positiveTrainPath)
        for i, testDataFile in enumerate(testDataFiles):
            if (testDataFile.split('.')[-1] != 'fbank'):
                testDataFiles.remove(testDataFile)
        for i, trainDataFile in enumerate(trainDataFiles):
            if (trainDataFile.split('.')[-1] != 'fbank'):
                trainDataFiles.remove(trainDataFile)
        return testDataFiles, trainDataFiles

    # Get a batch of positive training example
    def getTrainPositiveNextBatch(self,shuffle = True):
        self.resetDataDict()
        for i in range(self.dataFileBatchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0 # repeat the hole dataset again
                # return self.trainData
            print("Load from "+self.positiveTrainPath + self.trainDataFiles[self.currentTrainDataFile])
            result,label = self.util.fbankTransform(self.positiveTrainPath + self.trainDataFiles[self.currentTrainDataFile])
            self.currentTrainDataFile += 1
            self.trainData['data'] += result
            self.trainData['label'] += label
        self.trainData['data'],self.trainData['label'] = np.array(self.trainData['data']),np.array(self.trainData['label'])
        self.trainData = tf.data.Dataset.from_tensor_slices(self.trainData) # Covert raw data (dict) to tensorflwo Dataset type
        if (self.shuffle == True):
            self.trainData.shuffle(buffer_size=1000)
        return self.trainData

    # Get a batch of positive test example
    def getTestPositiveNextBatch(self):
        self.resetDataDict()
        for i in range(self.dataFileBatchSize):
            if(self.currentTrainDataFile >= len(self.testDataFiles)):
                self.currentTrainDataFile = 0
                # return self.testData
            # print("Load from "+self.positiveTestPath + self.testDataFiles[self.currentTestDataFile])
            result,label = self.util.fbankTransform(self.positiveTestPath + self.testDataFiles[self.currentTestDataFile])
            self.currentTestDataFile += 1
            self.testData['data'] += result
            self.testData['label'] += label
        self.testData['data'],self.testData['label'] = np.array(self.testData['data']),np.array(self.testData['label'])
        self.testData = tf.data.Dataset.from_tensor_slices(self.testData)
        if (self.shuffle == True):
            self.testData.shuffle(buffer_size=1000)
        return self.testData

    def resetDataDict(self):
        self.trainData = {"data": [], "label": []}
        self.testData = {"data": [], "label": []}

if __name__ == "__main__":
    dataloader = dataLoader(dataFileBatchSize=5)
    dataset = dataloader.getTrainPositiveNextBatch()
    iter = dataset.make_one_shot_iterator()
    data =  iter.get_next()
    with tf.Session() as sess:
        print(sess.run(data))
