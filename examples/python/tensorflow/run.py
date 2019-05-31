import os
import tensorflow as tf
import numpy as np
from blocksparse import BlocksparseMatMul

#data_files_path = tf.resource_loader.get_data_files_path()
#library_dir = '/home/philippe/development/triton/build/examples/python/tensorflow'
#module = tf.load_op_library(os.path.join(library_dir, 'libtf_blocksparse.so'))

hidden_size = 4096
block_size = 32
minibatch_size = 64

# Initialize
sparsity = np.random.randint(2, size=(hidden_size//block_size,hidden_size//block_size))
bsmm = BlocksparseMatMul(sparsity, block_size=block_size)
x = tf.placeholder(tf.float32, shape=[None, hidden_size])
w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)
y = bsmm(x, w)

# Run
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result = sess.run([y], feed_dict = {x: np.ones((minibatch_size,hidden_size), dtype='float32')})
print(result)
