# _*_ coding:utf-8 _*_
import os
import sys
import numpy as np
import matplotlib.pyplot as plt    # 导入python 绘图库
caffe_root = '//usr//local//caffe//caffe-master//'   # 设置caffe_root目录
sys.path.insert(0, caffe_root + 'python')   # 添加系统环境变量
import caffe
MODEL_FILE = '/usr/local/caffe/caffe-master/examples/mnist/lenet.prototxt'  # 指定LeNet网络定义文件
PRETRAINED = '/usr/local/caffe/caffe-master/examples/mnist/lenet_iter_10000.caffemodel' # 指定训练好的LeNet模型文件
# IMAGE_FILE = '/home/jacoob/workStation/pycharmProject/caffeExercises/mnistModel/mnist_test/train_9.bmp'# 待测试的图片文件路径
IMAGE_FILE = 'mnist_test/train_9.bmp'
#input_image = caffe.io.load_image(IMAGE_FILE, PRETRAINED)    # caffe 接口载入测试图片文件
input_image = caffe.io.load_image(IMAGE_FILE, False)
# print input_image
net = caffe.Classifier(MODEL_FILE, PRETRAINED)  # 载入LeNet分类器
# 预测图片进行分类，没有crop时， oversample过采样为false
prediction = net.predict([input_image], oversample=False)
# 设置为CPU模式
caffe.set_mode_cpu()
# 打印分类结果
print 'predicted class: ', prediction[0].argmax()



