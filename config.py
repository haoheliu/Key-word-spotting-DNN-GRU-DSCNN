import tensorflow as tf

class Config:
    localHost = True
    useDeepModel = False
    trainBatchSize = 100
    testBatchSize = 0
    leftFrames = 30
    shuffle = True
    rightFrames = 10
    learningRate = 0.000001
    decay_rate = 0.85
    numEpochs = 30
    w_smooth = 3
    w_max = 10
    maximumFrameNumbers = 1000

    base = 5

    if(localHost == True):
        positiveTestPath = "./data/Positive/test/"
        positiveTrainPath = "./data/Positive/train/"
        negativeTestPath = "./data/Negative/test/"
        negativeTrainPath = "./data/Negative/train/"
        offlineDataPath = "./offlineData/"
    else:
        positiveTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/test/"
        positiveTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/train/"
        negativeTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/negative/test/"
        negativeTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/negative/train/"
        offlineDataPath = "./offlineData/"

if __name__  == "__main__":
    Config.numEpochs -= 1
    print(Config.numEpochs)