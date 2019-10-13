import tensorflow as tf
meta_graph_path = './models/GRU/model.ckpt.meta'
load_mod = tf.train.import_meta_graph(meta_graph_path)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print("here"+str(variable_parameters))
    total_parameters += variable_parameters
print('The total number of parameters are: ', total_parameters)

# GRU 372483