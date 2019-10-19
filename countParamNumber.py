import tensorflow as tf
meta_graph_path = './models/DSCNN/model.ckpt.meta'
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

'''
(10, 4, 1, 172)
4
10
4
1
172
here6880
(172,)
1
172
here172
(172,)
1
172
here172
(3, 3, 172, 1)
4
3
3
172
1
here1548
(172,)
1
172
here172
(172,)
1
172
here172
(1, 1, 172, 172)
4
1
1
172
172
here29584
(172,)
1
172
here172
(172,)
1
172
here172
(3, 3, 172, 1)
4
3
3
172
1
here1548
(172,)
1
172
here172
(172,)
1
172
here172
(1, 1, 172, 172)
4
1
1
172
172
here29584
(172,)
1
172
here172
(172,)
1
172
here172
(3, 3, 172, 1)
4
3
3
172
1
here1548
(172,)
1
172
here172
(172,)
1
172
here172
(1, 1, 172, 172)
4
1
1
172
172
here29584
(172,)
1
172
here172
(172,)
1
172
here172
(3, 3, 172, 1)
4
3
3
172
1
here1548
(172,)
1
172
here172
(172,)
1
172
here172
(1, 1, 172, 172)
4
1
1
172
172
here29584
(172,)
1
172
here172
(172,)
1
172
here172
(172, 3)
2
172
3
here516
(3,)
1
3
here3
The total number of parameters are:  135023
'''