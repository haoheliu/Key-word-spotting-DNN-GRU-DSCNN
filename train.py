from model import Model
from dataLoader import dataLoader
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
num_epochs = 201
batchSize = 1000
dataFileBatchSize=10

def lossFunc(model,inputs,targets):
    inputs = tf.reshape(inputs, shape=[batchSize, 1640])
    outputs = tf.nn.softmax(model(inputs))
    targets = dataloader.util.label2tensor(targets)
    output = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels = targets)
    output = tf.reduce_sum(output)
    return output

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = lossFunc(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# with tf.device('/device:GPU:0'):
dnnModel = Model()
dataloader = dataLoader(dataFileBatchSize=dataFileBatchSize)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step()
train_loss_results = []
train_accuracy_results = []

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    dataset = dataloader.getTrainPositiveNextBatch() # Get a batch of data
    dataset = dataset.batch(batchSize) # Divided this dataset into several batches
    iter =  dataset.make_one_shot_iterator() # Take one batch a time
    # try:
    batch = iter.get_next()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(batch['data']).shape,sess.run(batch['label']).shape)
        loss_value, grads = grad(dnnModel.model,batch['data'], batch['label'].eval())
        optimizer.apply_gradients(zip(grads, dnnModel.model.variables), global_step)
        print("loss ",sess.run(loss_value))
    # except:

    #     continue