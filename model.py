import tensorflow as tf
class Model:
    def __init__(self):
        self.model = None
        self.createModel()

    def createModel(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(1640,)),  # input shape required
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(3)
        ])
if __name__ == "__main__":
    m = Model()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(m.model.variables[1]))