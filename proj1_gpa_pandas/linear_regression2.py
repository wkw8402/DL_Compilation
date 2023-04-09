import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a = tf.Variable(0.1) #randomize
b = tf.Variable(0.1)

def loss_function(a, b):
    predict_y = train_x * a + b
    return tf.keras.losses.mse(train_y, predict_y)

opt = tf.keras.optimizers.Adam(learning_rate=0.001) #change learning rate, a hyperparameter, unil we get good result

for i in range(9000):
    opt.minimize(loss = lambda: loss_function(a, b), var_list=[a,b])
    print(a.numpy(), b.numpy())

# 1. make model
# 2. choose optimizer, loss function
# 3. update vairable through gradient descent