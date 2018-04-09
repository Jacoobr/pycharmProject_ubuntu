#!/usr/bin/env python
# coding:utf-8
"""
@version: 
@author: xiaojianli
@contact:xiaoli_lixj@163.com
@software: PyCharm
@time:  下午8:58
"""
import tensorflow as tf
import time

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                      [2]])

multi = tf.matmul(matrix1, matrix2) # tf matrixs multiply ; np.dot(m1, m2);

# ## method 1
# sess = tf.Session()
# res = sess.run(multi)
# print(res)
# sess.close()

# ## method 2
with tf.Session() as sess:
    tic = time.time()
    res = sess.run(multi)
    toc = time.time()
    print(res, (toc-tic)*1000)


