import tensorflow as tf
import os

from sklearn.utils import shuffle
from model import Model
from dataLoader import dataLoader
from config import Config

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto(allow_soft_placement=True) #log_device_placement=True
config.gpu_options.allow_growth = True
totalEpoches = Config.numEpochs

print("Construct model...")
dnnModel = Model()
print("Construct dataLoader...")
dataloader = dataLoader()
batch = tf.placeholder(tf.float32,shape = (None,40*(Config.leftFrames+Config.rightFrames+1)),name = 'batch_input')
label = tf.placeholder(tf.float32,shape = (None,3),name="label_input")
Loss,_ = dnnModel.lossFunc_CrossEntropy(batch, label)
saver = tf.train.Saver()
print("Construct optimizer...")

with tf.name_scope("modelOptimizer"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(Config.learningRate,
                                               global_step=global_step,
                                               decay_steps=int(10000/Config.trainBatchSize),
                                               decay_rate=Config.decay_rate)
    trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate,name="gradient_optimizer").minimize(Loss,global_step=global_step)

print("Start Training Session...")
if (not os.path.exists("./model/" )):
    os.mkdir("./model/" )
if (not os.path.exists("./DeepModel/" )):
    os.mkdir("./DeepModel/" )

with tf.Session(config=config) as sess:
    print("Initialize variables...")
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    # writer.close()
    while(not Config.numEpochs == 0):
        currentEpoch = Config.numEpochs
        # if(Config.numEpochs % 1 == 0):
        #     print("Start testing... ", end="")
        #     dataloader.visualizaPositiveDataFiles(dataloader.testDataFiles,sess,dnnModel.model)
        if(Config.useDeepModel == True):
            saver.save(sess,"./DeepModel/model.ckpt")
        else:
            saver.save(sess, "./Model/model.ckpt")
        print("done")
        exit(0)
        while(1):
            batchTrain,labelTrain = dataloader.getTrainNextBatch() # Get a batch of data
            batchTrain, labelTrain = shuffle(batchTrain,labelTrain)
            if(not currentEpoch == Config.numEpochs):
                break
            sess.run(trainStep,feed_dict={batch:batchTrain,label:labelTrain})
        print("[EPOCH " + str(totalEpoches - Config.numEpochs+1), "]", "lr: ", sess.run(learning_rate))

