import tensorflow as tf

tensor = tf.constant( [3,4,5] )
tensor2 = tf.constant( [6.0,7,8], tf.float32 ) #dtype=float32

print(tensor + tensor2)
print(tf.add(tensor, tensor2))

tensor3 = tf.constant( [ [1,2], 
                         [3,4] ] )
print(tensor3.shape)

tensor4 = tf.zeros( [2,2,3] ) #shape read from right to left
print(tensor4)

w = tf.Variable(1.0)
print(w.numpy()) #.numpy() returns value

w.assign(2)
print(w)


