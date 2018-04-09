# coding:utf-8

"""
tensorflow基础知识学习
"""
import tensorflow as tf
import numpy as np
"""
1.获取变量作用域
"""
##code
'''
with tf.variable_scope("foo") as foo_scope:
    v1 = tf.get_variable("v", [1], dtype=tf.float32)  # define tf variable
    assert foo_scope.name == "foo"
with tf.variable_scope("foo2"):
    with tf.variable_scope("foo3") as other_scope:
        assert other_scope.name == "foo2/foo3"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo"
# 如果开启一个变量作用域时使用之前的variable_scope， 则会先跳过当前作用域，保存预先存在的作用域不变
with tf.variable_scope(foo_scope):
    v2 = tf.get_variable("w", [1], dtype=tf.float32)
'''
"""
2. variable_scope的初始化
"""
'''
# 声明变量作用域并用常量初始化
with tf.variable_scope("foo", initializer=tf.constant_initializer(0.3)):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(0.4)) # alter the initialized value
    v2 = 1.0 + v
    # 被作用域初始化器初始化


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("the value of v2.eval() is:", v2.eval())  # [1.4]
    assert v.eval() == 0.4
    assert v2.eval() == 1.4
    assert v2.op.name == "foo/add"
'''
"""
3. Graph
"""
'''
metrix1 = tf.constant([[2., 3.],
                       [1., 4.]])
metrix2 = tf.constant([[1., 2.],
                       [3., 4]])
# define operation
product = tf.matmul(metrix1, metrix2)

# init variation
init = tf.initialize_all_variables()

# Session
with tf.Session() as sess:
    sess.run(init)
    res = sess.run([product]) # start session
    print(res)
'''
"""
Pooling function
"""
# tf.nn.avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None) 计算池化区域元素的平均值
# value:4 dimensions tensor ->[batch, height, width, channels]
# ksize: 一个长度不小于4的整形数组。每一位上的数值分别对应待卷积图像毎一维池化操作的矩阵窗口大小
# strides:一个长度不小于4的整形数组。每一位数值分别表示在待池化操作的图像上的毎一维上滑动的步长
# padding: 取值为SAME or VALID; SAME表示池化后的图像大小不变，VALID则池化后的图像大小发生变化
# data_format: 'NHWC'代表输入tensor维度的顺序，N -> 个数，H -> 高度， W -> 宽度， C -> tensor通道数（一般指RGB 3通道图像或单通道灰度图像）
# [name]: 改池化操作的名称
# return: tensor, 数据类型和value相同
'''
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=tf.float32)   #batch:10 height,width: 6 channels: 3
filter_ = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=tf.float32)  #height,width: 2  input_channel: 3 output_channel: 10 (卷积核个数)

y = tf.nn.conv2d(input=input_data, filter=filter_, strides=[1, 1, 1, 1], padding="SAME")
print(y)
res = tf.nn.avg_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
print(res)
print([10, 6, 6, 10]-[1, 2, 2, 1]+1)
'''


"""
模型的存储和加载
定义一个神经网络，结构为：两个全连接层 + 一个输出层
"""
##加载数据及模型定义
#加载数据模型包括：1.建立train.Saver()保存变量； 2. 制定保存位置，以什么文件保存（一般扩展名为.ckpt）
import os
from tensorflow.examples.tutorials.mnist import input_data

#加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #one_hot用于描述一张图片的标签（0-9），one_hot_vectors向量除了标签的数字处是1,其余均为0
trX, trY, validX, validY, teX, teY = mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, mnist.test.images, mnist.test.labels
print("trX, trY, validX, validY, teX, teY 的大小为：")
print(trX.shape, trY.shape, validX.shape, validY.shape, teX.shape, teY)

X = tf.placeholder(tf.float32, [None, 784])    #图片大小：28*28=784. Its value must be fed using the `feed_dict` optional argument to `Session.run()`,`Tensor.eval()`, or `Operation.run()`.
Y = tf.placeholder(tf.float32, [None, 10])     #one-hot_vectors


#定义权重函数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))    #stddev: he standard deviation of the normal distribution. default is 1.0


#初始化权重
w_h = init_weights([784, 625])
w_h2 = init_weights([625, 625])
w_o = init_weights([625, 10])


#定义模型
def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    #第一个全连接层
    X = tf.nn.dropout(X, p_keep_input)  #定义池化函数
    h = tf.nn.relu(tf.matmul(X, w_h))    #定义激活函数
    h = tf.nn.dropout(h, p_keep_hidden)

    #第二个全连接层
    h2 = tf.nn.relu(tf.matmul(h, w_h2))  #tf.matmul()：Multiplies matrix `a` by matrix `b`, producing `a` * `b`
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)   #输出预测值


#生成网络模型，得到预测值
p_keep_input = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)


#定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  #使用softmax_cross_entropy分类器
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)    #使用RMSPropOptimizer优化器,最小化cost,　learning_rate = 0.001, decay = 0.9
predict_op = tf.argmax(py_x, 1) #返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。


#训练模型并存储模型
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):    #set the path where the parameter file stored
    os.makedirs(ckpt_dir)


#定义计数器变量,初始化为0,trainable=False,则该变量不会被训练
global_step = tf.Variable(0, name="global_step_train", trainable=False)

##tf.train.Saver() 保存和提取变量,定义在其后的变量则不会被存储
#在声明所有变量后调用 tf.train.Saver()
saver = tf.train.Saver()

non_stored_variable = tf.Variable(777)  #该值不会被存储


#训练模型并存储
with tf.Session() as sess:
    tf.initialize_all_variables().run() #初始化变量 or sess.run(init = tf.initialize_all_variables())

    start = global_step.eval()    #return the value of global_step_train
    print("the training from: ", start)

    for i in range(start, 100): #set the epoch =100
        #128为bitch_size
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_input: 0.8, p_keep_hidden: 0.5})
        global_step.assign(i).eval()  #update the counter
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)   #store the model




























