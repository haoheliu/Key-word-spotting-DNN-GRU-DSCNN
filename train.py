from model import Model
from dataLoader import dataLoader
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

dnnModel = Model()
dataloader = dataLoader(dataFileBatchSize=2)

def lossFunc(model,inputs,targets):
    inputs = tf.reshape(tf.cast(inputs, dtype=tf.float32), shape=[1, 1640])
    outputs = tf.nn.softmax(model(inputs))
    targets = dataloader.util.label2tensor(targets)
    output = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels = targets)
    return output

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = lossFunc(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.train.get_or_create_global_step()


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     while(True):
#         data = iter.get_next()
#         loss_value, grads = grad(dnnModel.model, data, sess.run(data['label']))
#         print("Step: {}, Initial Loss: {}".format(sess.run(global_step),
#                                                   sess.run(loss_value)))
#         optimizer.apply_gradients(zip(grads, dnnModel.model.variables), global_step)
#         after = lossFunc(dnnModel.model, data, sess.run(data['label']))
#         print("Step: {},         Loss: {}".format(sess.run(global_step),
#                                                   sess.run(after)))

train_loss_results = []
train_accuracy_results = []

num_epochs = 201

with tf.Session() as sess:
    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        dataset = dataloader.getTrainPositiveNextBatch() # Get a batch of data
        iter = dataset.make_one_shot_iterator()
        try:
            counter = 0
            while(True):
                if(counter % 100 == 1):
                    print("here counter: " ,counter)
                counter += 1
                data = iter.get_next()
                label = sess.run(data['label'])
                loss_value, grads = grad(dnnModel.model,data['data'], label)
                optimizer.apply_gradients(zip(grads, dnnModel.model.variables), global_step)
                epoch_loss_avg(loss_value)
                # epoch_accuracy(tf.argmax(dnnModel.model(data['data']), axis=1, output_type=tf.int32),label )
        except:
            print("end",sess.run(epoch_loss_avg.result()))

        # train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results.append(epoch_accuracy.result())
        # print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                         epoch_loss_avg.result(),
        #                                                         epoch_accuracy.result()))