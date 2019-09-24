from util import Util
import os

class dataLoader:
    def __init__(self,dataFileBatchSize = 2):
        self.util = Util()
        self.dataFileBatchSize = dataFileBatchSize
        self.trainData = {"data": [], "label": []}
        self.testData = {"data": [], "label": []}
        self.testDataFiles, self.trainDataFiles = self.constructPositiveDataSet()
        self.currentTestDataFile, self.currentTrainDataFile = 0,0
        self.positiveTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/test/"
        self.positiveTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/train/"

    def constructPositiveDataSet(self,):
        testDataFiles, trainDataFiles = os.listdir(self.positiveTestPath), os.listdir(self.positiveTrainPath)
        for i, testDataFile in enumerate(testDataFiles):
            if (testDataFile.split('.')[-1] != 'fbank'):
                testDataFiles.remove(testDataFile)
        for i, trainDataFile in enumerate(trainDataFiles):
            if (trainDataFile.split('.')[-1] != 'fbank'):
                trainDataFiles.remove(trainDataFile)
        return testDataFiles, trainDataFiles

    def getTrainPositiveNextBatch(self):
        self.resetDataDict()
        for i in range(self.dataFileBatchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                return self.trainData
            result,label = self.util.fbankTransform(self.positiveTrainPath + self.trainDataFiles[self.currentTrainDataFile])
            self.currentTrainDataFile += 1
            self.trainData['data'] += result
            self.trainData['label'] += label
        return self.trainData

    def resetDataDict(self):
        self.trainData = {"data": [], "label": []}
        self.testData = {"data": [], "label": []}

if __name__ == "__main__":
    dataloader = dataLoader()
    result = dataloader.getTrainPositiveNextBatch()
