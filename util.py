import fbankreader3
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import wave

class Util:
	def __init__(self,sampleRate = 160):
		self.sampleRate = sampleRate  # per 0.01 second
		self.bundary = self.loadPositivePlace()

	# Combine each frame's feature with left and right frames'
	def fbankTransform(self,fPath = "positive_00011.fbank",left = 30,right = 10):
		raw = fbankreader3.HTKFeat_read(fPath)
		raw = raw.getall().tolist()
		result = []
		fname = fPath.split('.')[-2].split('/')[-1]  # e.g. positive_00011
		label = [0] * len(raw)
		for i in range(len(raw)): # Initialize a collection
			result.append([])
		raw = [raw[0]]*left+raw+[raw[-1]]*right
		for i in range(0,len(result)):
			base = i+left 					 # the frame
			for left_i in range(left,0,-1):  # front 30 frames
				result[i] += raw[base-left_i]
			result[i] += raw[base]
			for right_i in range(1,right+1):
				result[i] += raw[base+right_i]
		for i in range(0, len(result)):
			if(self.isFirstKeyWord(fname,i)):
				label[i] = 1
			elif(self.isSecondKeyWord(fname,i)):
				label[i]  = 2
		return result,label

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

	def label2tensor(self,label):
		# if(label < 0 or label > 2):
		# 	raise("label value should within 0 - 2")
		retVal = []
		for each in label:
			newTensor = [0,0,0]
			newTensor[each] = 1
			retVal.append(newTensor)
		return tf.convert_to_tensor(np.array(retVal))

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
	util.plotWave(fname="positive_00013")




