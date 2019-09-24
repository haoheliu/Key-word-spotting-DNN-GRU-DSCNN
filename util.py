import fbankreader
import os
import numpy as np
import pickle

class Util:
	def __init__(self):
		self.sampleRate = 160  # per 0.01 second
		self.bundary = self.loadPositivePlace()
		self.trainData = {"data":[],"label":[]}
		self.testData = {"data":[],"label":[]}
		self.dataBatchSize = 10
	# Combine each frame's feature with left and right frames'
	def fbankTransform(self,fPath = "positive_00011.fbank",left = 30,right = 10):
		raw = fbankreader.HTKFeat_read(fPath)
		raw = raw.getall().tolist()
		result = []
		fname = fPath.split('.')[-2].split('/')[-1]  # e.g. positive_00011
		label = [0] * len(raw)
		for i in range(len(raw)): # Initialize a collection
			result.append([])
		raw = [raw[0]]*left+raw+[raw[-1]]*right
		for i in range(0,len(result)):
			for left_i in range(1,left+1):
				result[i] += raw[i-left_i]
			result[i] += raw[i+left]
			for right_i in range(1,right+1):
				result[i] += raw[i+right_i]
		for i in range(0, len(result)):
			if(self.isFirstKeyWord(fname,i)):
				label[i] = 1
			elif(self.isSecondKeyWord(fname,i)):
				label[i]  = 2
		return result,label

	def resetDataDict(self):
		self.trainData = {"data": [], "label": []}
		self.testData = {"data": [], "label": []}

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

	def isFirstKeyWord(self,fname,num):
		return num >= self.bundary[fname][0] and num <= self.bundary[fname][1]

	def isSecondKeyWord(self,fname,num):
		return num >= self.bundary[fname][2] and num <= self.bundary[fname][3]

	def constructPositiveDataSet(self,positiveTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/test/" ,\
									positiveTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/train/"):
		testDataFiles, trainDataFiles = os.listdir(positiveTestPath), os.listdir(positiveTrainPath)
		for i,testDataFile in enumerate(testDataFiles):
			if (testDataFile.split('.')[-1] != 'fbank'):
				testDataFiles.remove(testDataFile)
			# data,labels = self.fbankTransform(positiveTestPath+testDataFile)
			# self.testData['data'].append(data)
			# self.testData['label'].append(labels)
			# i += 1
			# if (i % self.dataBatchSize == 0):
			# 	self.savePkl("./Data/testPositive_"+str(i/self.dataBatchSize)+".pkl", self.testData)
			# 	self.resetDataDict()
			# 	break
		for i,trainDataFile in enumerate(trainDataFiles):
			if (trainDataFile.split('.')[-1] != 'fbank'):
				trainDataFiles.remove(trainDataFile)
			# data,labels = self.fbankTransform(positiveTrainPath+trainDataFile)
			# self.trainData['data'].append(data)
			# self.trainData['label'].append(labels)
			# i += 1
			# if(i%self.dataBatchSize == 0):
			# 	self.savePkl("./Data/trainPositive_"+str(i/self.dataBatchSize)+".pkl", self.trainData)
			# 	self.resetDataDict()
			# 	break
	def savePkl(self,fname,obj):
		print("Save ",fname)
		with open(fname,'wb') as f:
			pickle.dump(obj,f)

if __name__ == "__main__":
	util = Util()
	util.constructPositiveDataSet()




