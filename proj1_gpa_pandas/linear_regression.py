import tensorflow as tf

height = 170
shoe = 260
#shoe = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_function(): #mean squared error
    prediction = height * a + b
    return tf.square(260 - prediction )

opt = tf.keras.optimizers.Adam(learning_rate=0.1) 

for i in range(300):
    opt.minimize(loss = loss_function, var_list=[a,b])
    print(a.numpy(), b.numpy())

