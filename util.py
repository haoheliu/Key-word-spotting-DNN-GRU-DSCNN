import fbankreader3
import pickle

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

	def savePkl(self,fname,obj):
		print("Save ",fname)
		with open(fname,'wb') as f:
			pickle.dump(obj,f)

	def loadPkl(self,fname):
		print("Load ",fname)
		with open(fname,'rb') as f:
			result = pickle.load(f)
		return result

if __name__ == "__main__":
	pass




