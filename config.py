import tensorflow as tf

class Config:
    modelName = "DSCNN" # "GRU" "DNN_6_512" "DNN_3_128" "DSCNN"
    lossFunc = "seqLoss" # "Paper" "crossEntropy"
    trainBatchSize = 10
    testBatchSize = 10
    leftFrames = 15
    shuffle = True
    rightFrames = 5
    learningRate = 0.000002
    decay_rate = 0.895
    numEpochs = 60
    w_smooth = 5
    w_max = 70
    maximumFrameNumbers = 1300 # Max: 1259
    visualizeTestData = False
    useTensorBoard = True
    testMode = False
    saveModel = True
    drowROC = True
    enableVisualize = visualizeTestData or drowROC
    positiveTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/test/"
    positiveTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/positive/train/"
    negativeTestPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/negative/test/"
    negativeTrainPath = "/home/disk2/internship_anytime/aslp_hotword_data/aslp_wake_up_word_data/data/negative/train/"
    offlineDataPath = "./offlineData/"
    logDir = "./log"

    # DSCNN
    model_settings = {
        'stack_frame_number': 1+leftFrames+rightFrames,
        'feature_number': 40,
        'label_count': 3,
    }
    model_size_info = [5,172,10,4,2,1,172,3,3,2,2,172,3,3,1,1,172,3,3,1,1,172,3,3,1,1]
    is_training = True

    featureLength = model_settings['stack_frame_number'] * model_settings['feature_number']

if __name__  == "__main__":
    Config.numEpochs -= 1
    print(Config.numEpochs)