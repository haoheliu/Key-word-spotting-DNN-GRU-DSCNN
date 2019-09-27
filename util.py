import fbankreader3
import pickle
import wave
import os

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from sklearn.metrics import roc_curve, auc
from config import Config

class Util:
    def __init__(self,sampleRate = 160):
        self.sampleRate = sampleRate  # per 0.01 second
        self.bundary = self.loadPositivePlace()
        self.testPositiveDataFiles, self.trainPositiveDataFiles = self.constructPositiveDataSet()
        self.testNegativeDataFiles, self.trainNegativeDataFiles = self.constructNegativeDataset()

    def splitFileName(self,fname):
        return fname.split('.')[-2].split('/')[-1]

    # Combine each frame's feature with left and right frames'
    def fbankTransform(self,fPath = "positive_00011.fbank",save = True):
        raw = fbankreader3.HTKFeat_read(fPath)
        raw = raw.getall().tolist()
        frameLength = len(raw)
        result = np.empty(shape=[0,(Config.leftFrames+Config.rightFrames+1)*40])
        fname = self.splitFileName(fPath)  # e.g. positive_00011
        label = np.empty(shape=[0,3])
        raw = np.array([raw[0]]*Config.leftFrames+raw+[raw[-1]]*Config.rightFrames) # This trick can make algorithm more efficient
        for i in range(0,frameLength):
            base = i + Config.leftFrames
            temp = raw[i:base + Config.rightFrames + 1].reshape((1,1640)) # This can outperform np.append
            result = np.concatenate((result, temp),axis=0)
        for i in range(0, len(result)):
            if(self.isFirstKeyWord(fname,i)):
                label = np.append(label,np.array([[0,1,0]]),axis=0)
            elif(self.isSecondKeyWord(fname,i)):
                label = np.append(label,np.array([[0,0,1]]),axis=0)
            else:
                label = np.append(label, np.array([[1, 0, 0]]),axis=0)
        if(save == True):
                np.save(Config.offlineDataPath + fname+"_data.npy",result)
                np.save(Config.offlineDataPath + fname + "_label.npy", label)
        return result,label

    def fbankBatchTransform(self,dataFiles,fPath):
        print("Construct offline test data from " + fPath + "...")
        counter = 0
        for fname in dataFiles:
            counter += 1
            if (counter % 50 == 0):
                print(str(counter) + " files Finished"+"; Totally "+str(len(dataFiles))+" files")
            try:
                self.fbankTransform(fPath=fPath + fname)
            except:
                print("Error while transforming " + fPath + fname)
                continue

    def constructOfflineData(self):
        if(not os.path.exists("offlineData")):
            os.mkdir("offlineData")
        self.fbankBatchTransform(self.testPositiveDataFiles,Config.positiveTestPath)
        self.fbankBatchTransform(self.trainPositiveDataFiles,Config.positiveTrainPath)
        self.fbankBatchTransform(self.testNegativeDataFiles,Config.negativeTestPath)
        self.fbankBatchTransform(self.trainNegativeDataFiles,Config.negativeTrainPath)

    def plotWave(self,fname = "positive_00011"):
        util = Util()
        positivePlace = util.loadPositiveBitsPlace()
        f = wave.open("./"+fname+".WAV", 'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.fromstring(str_data, dtype=np.short)
        time = np.arange(0, nframes) * (1.0 / framerate)
        # wave_data.shape = -1, 2  #这里可以分出左右声道
        # wave_data = wave_data.T
        bitsLabel = [0] * len(time)
        for i in range(positivePlace[fname][0], positivePlace[fname][1]):
            bitsLabel[i] = 1
        for i in range(positivePlace[fname][2], positivePlace[fname][3]):
            bitsLabel[i] = 2
        pl.subplot(211)
        pl.plot(time, wave_data)
        pl.subplot(212)
        pl.plot(time, bitsLabel, c="g")
        pl.xlabel("time (seconds)")
        pl.show()

    def plotRoc(self,labels, predict_prob,show = True):
        false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
        roc_auc=auc(false_positive_rate, true_positive_rate)
        if(show == True):
            plt.title('ROC')
            plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.ylabel('TPR')
            plt.xlabel('FPR')
            plt.show()
        return roc_auc

    # Load by frames
    def loadPositivePlace(self,fPath = "./positiveKeywordPosition.txt"):
        buffer = []
        with open(fPath,'r') as f:
            buffer = f.readlines()
        result = {}
        for i in range(len(buffer)):
            temp = buffer[i].strip().split()
            result[temp[0]] = [int(int(each)/self.sampleRate) for each in temp[1:]]  # Ignore boundary frame with "int()"
        return result

    # Load by sample points
    def loadPositiveBitsPlace(self,fPath = "./positiveKeywordPosition.txt"):
        buffer = []
        with open(fPath,'r') as f:
            buffer = f.readlines()
        result = {}
        for i in range(len(buffer)):
            temp = buffer[i].strip().split()
            result[temp[0]] = [int(each) for each in temp[1:]]  # Ignore boundary frame with "int()"
        return result

    def isFirstKeyWord(self,fname,num):
        if(fname not in self.bundary.keys()):
            return False
        return num >= self.bundary[fname][0] and num <= self.bundary[fname][1]

    def isSecondKeyWord(self,fname,num):
        if (fname not in self.bundary.keys()):
            return False
        return num >= self.bundary[fname][2] and num <= self.bundary[fname][3]

    def savePkl(self,fname,obj):
        print("Save ",fname)
        with open(fname,'wb') as f:
            pickle.dump(obj,f)

    def loadPkl(self,fname):
        print("Load ",fname)
        with open(fname,'rb') as f:
            result = pickle.load(f)
        return result

    # List All the Positive test&train files
    def constructPositiveDataSet(self):
        testDataFiles, trainDataFiles = os.listdir(Config.positiveTestPath), os.listdir(Config.positiveTrainPath)
        for i, testDataFile in enumerate(testDataFiles):
            if (testDataFile.split('.')[-1] != 'fbank'):
                testDataFiles.remove(testDataFile)
        for i, trainDataFile in enumerate(trainDataFiles):
            if (trainDataFile.split('.')[-1] != 'fbank'):
                trainDataFiles.remove(trainDataFile)
        return testDataFiles, trainDataFiles

    def constructNegativeDataset(self):
        testDataFiles, trainDataFiles = os.listdir(Config.negativeTestPath), os.listdir(Config.negativeTrainPath)
        for i, testDataFile in enumerate(testDataFiles):
            if (testDataFile.split('.')[-1] != 'fbank'):
                testDataFiles.remove(testDataFile)
        for i, trainDataFile in enumerate(trainDataFiles):
            if (trainDataFile.split('.')[-1] != 'fbank'):
                trainDataFiles.remove(trainDataFile)
        return testDataFiles, trainDataFiles

if __name__ == "__main__":
    util = Util()
    util.constructOfflineData()  # 离线构建数据集





