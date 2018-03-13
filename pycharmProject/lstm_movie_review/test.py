#coding:utf-8
'''
softmax:S_{i} = e^(v_{i})/(/sum_{j} e^(v_{j}))
Loss: L_i = -log(e^(v_{i})/(/sum_{j} e^(v_{j}))) 即：-log(S_{i})
'''
# caculate vector [1, 5, 3]
import numpy as np
cac = [1, 5, 3]
res_softmax = [np.exp(vi)/np.sum(np.exp(cac)) for vi in cac]  #softmax function
sum_softmax = np.sum(res_softmax) # should be 1.0
#Li = np.log(res_softmax)    # Loss为交叉商
pass
