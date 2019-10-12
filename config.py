import tensorflow as tf

class Config:
    localHost = False
    modelName = "GRU" # "GRU" "DNN_6_512" "DNN_3_128"
    lossFunc = "seqLoss" # "Paper" "crossEntropy"
    trainBatchSize = 100
    testBatchSize = 100
    leftFrames = 15
    shuffle = True
    rightFrames = 5
    learningRate = 0.001
    decay_rate = 0.5
    numEpochs = 2
    w_smooth = 3
    w_max = 10
    maximumFrameNumbers = 1300 # Max: 1259

    useTensorBoard = True
    testMode = False
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