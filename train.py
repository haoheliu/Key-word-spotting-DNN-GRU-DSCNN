from model import Model
from dataLoader import dataLoader
import tensorflow as tf
import os
from config import Config

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

dnnModel = Model()
dataloader = dataLoader(dataFileBatchSize=Config.batchSize)

batch = tf.placeholder(tf.float32,shape = (None,40*(Config.leftFrames+Config.rightFrames+1)),name = 'batch_input')
label = tf.placeholder(tf.float32,shape = (None,3),name="label")

def lossFunc(model,inputs,targets):
    outputs0 = tf.nn.softmax(model(tf.convert_to_tensor(inputs,dtype=tf.float32)))
    output1 = tf.nn.softmax_cross_entropy_with_logits(logits=outputs0,labels = targets)
    output = tf.reduce_sum(output1)
    return output

testloss = []

Loss = lossFunc(dnnModel.model, batch, label)

trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate).minimize(Loss)

testBatch,testLabel = dataloader.getTestPositiveNextBatch()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(Config.numEpochs):
        counter = 0
        testLoss = lossFunc(dnnModel.model, testBatch, testLabel)
        print("Test loss: ", sess.run(testLoss))
        while(True):
            if(counter % 10 == 0):
                print(counter)
            counter += 1
            batchTrain,labelTrain = dataloader.getTrainPositiveNextBatch() # Get a batch of data
            if(batchTrain.shape == (0,) and labelTrain.shape == (0,)):
                break
            dnnModel.model.save_weights(Config.modelPath1)
            sess.run(trainStep,feed_dict={batch:batchTrain,label:labelTrain})
            dnnModel.model.save_weights(Config.modelPath2)


