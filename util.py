import fbankreader3
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import wave
from config import Config
import os

class Util:
    def __init__(self,sampleRate = 160):
        self.sampleRate = sampleRate  # per 0.01 second
        self.bundary = self.loadPositivePlace()
        self.testDataFiles, self.trainDataFiles  = self.constructPositiveDataSet()

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
        raw = np.array([raw[0]]*Config.leftFrames+raw+[raw[-1]]*Config.rightFrames)
        for i in range(0,frameLength):
            base = i + Config.leftFrames
            temp = raw[i:base + Config.rightFrames + 1].reshape((1,1640))
            result = np.concatenate((result, temp),axis=0)
        for i in range(0, len(result)):
            if(self.isFirstKeyWord(fname,i)):
                label = np.append(label,np.array([[0,1,0]]),axis=0)
            elif(self.isSecondKeyWord(fname,i)):
                label = np.append(label,np.array([[0,0,1]]),axis=0)
            else:
                label = np.append(label, np.array([[1, 0, 0]]),axis=0)
        if(save == True):
            np.save("./offlineData/"+fname+"_data.npy",result)
            np.save("./offlineData/" + fname + "_label.npy", label)
        return result,label

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

    # Load
    def loadPositivePlace(self,fPath = "./positiveKeywordPosition.txt"):
        buffer = []
        with open(fPath,'r') as f:
            buffer = f.readlines()
        result = {}
        for i in range(len(buffer)):
            temp = buffer[i].strip().split()
            result[temp[0]] = [int(int(each)/self.sampleRate) for each in temp[1:]]  # Ignore boundary frame with "int()"
        return result

    # Load
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
        return num >= self.bundary[fname][0] and num <= self.bundary[fname][1]

    def isSecondKeyWord(self,fname,num):
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

    def constructOfflineData(self):
        if(not os.path.exists("offlineData")):
            os.mkdir("offlineData")
        print("Construct offline test data from "+Config.positiveTestPath+"...")
        counter = 0
        for fname in self.testDataFiles:
            counter += 1
            if(counter %50 == 0):
                print(str(counter)+" testfiles Finished")
            try:
                self.fbankTransform(fPath=Config.positiveTestPath+fname)
            except:
                print("Error while transforming "+fname)
                continue
        print("Construct offline train data from " + Config.positiveTrainPath + "...")
        counter = 0
        for fname in self.trainDataFiles:
            counter += 1
            if(counter %50 == 0):
                print(str(counter)+" trainfiles Finished")
            try:
                self.fbankTransform(fPath=Config.positiveTrainPath + fname)
            except:
                print("Error while transforming " + fname)
                continue

    def plotWave(self,fname = "positive_00001"):
        util = Util()
        positivePlace = util.loadPositiveBitsPlace()
        f = wave.open("./"+fname+".WAV", 'rb')
        # 读取格式信息
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # 读取波形数据
        str_data = f.readframes(nframes)
        f.close()
        # 将波形数据转换为数组
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

if __name__ == "__main__":
    util = Util()
    util.constructOfflineData()





