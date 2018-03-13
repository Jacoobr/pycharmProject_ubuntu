# _*_ coding:utf-8 _*_
import numpy as np
import struct
from PIL import Image   # 载入图片库
from skimage import io, data

# 指定压缩格式的测试样本库
filename = '//usr//local//caffe//caffe-master//data/mnist//t10k-images-idx3-ubyte'
# 以bin格式打开
binfile = open(filename, 'rb')
# 读文件所有内容到缓存buf
buf = binfile.read()
# 指定类型读数据，得到图片总数
index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

for image in range(0, numImages/100):   # 循环读取图片
    im = struct.unpack_from('>784B', buf, index)  # 读取一个图片
    index += struct.calcsize('>784B')

    im = np.array(im, dtype='uint8')
    im = im.reshape(28, 28)     # 更改图片的大小为 28*28
    #fig = plt.figure()
    #plotwindow = fig.add_subplot(111)
    #plt.imshow(im , cmap='gray')
    #plt.show()
    im = Image.fromarray(im)
    im.save('mnist_test/train_%s.bmp' % image, 'bmp',)   # 保存转换后的图片
#im = Image.open('mnist_test/train_9.bmp').convert('GR')
#im.save('train_%s.bmp' % image, 'bmp',)



