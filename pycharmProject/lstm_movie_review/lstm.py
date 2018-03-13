#coding:utf-8

'''
构建语义分析器
'''
from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict  #导入有序字典类
import sys
import time

import numpy as np
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams   #theano random number generator

import imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

SEED = 123  #　设置随机种子
np.random.seed(SEED)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)    #asarray　将数据转换为ndarry且始终保持最新copy

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    #对数据随机排列, n为总数据个数,minibatch_size为块大小,总划分出n/minibatch_size
    #+1个列表,拼在一起构成一个二重列表
    #返回一个zip(batch索引, batch内容),每份batch是一个列表,包含句子和索引
    #会根据句子的索引 拼出x
    idx_list = np.arange(n, dtype="int32") #返回一个0-n的一个数组

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0 #设置起始索引下标
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start+minibatch_size])  #将n//minibatch_size个数据快装入minibatches
        minibatch_start += minibatch_size
    #将剩下的数据装入
    if(minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    '''
    zip(seq1 [, seq2 [...]]) -> [(seq1[0], seq2[0] ...), (...)]　－> Return a list of tuples, where each tuple contains the i-th element
    from each of the argument sequences.  The returned list is truncated(截短)
    in length to the length of the shortest argument sequence.
    '''
    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]  # (datasets is a dictionary) 返回样本及对应的标签


def zipp(params, tparams):
    '''
    重新加载模型时需要gpu资源
    '''
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    '''
    当解压模型时需要gpu资源
    :param zipped:
    :return:
    '''
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    '''
    使用dropout规则化
    :param state_before:
    :param use_noise:
    :param trng:
    :return:
    '''
    proj = tensor.switch(use_noise,(state_before * trng.binomial(state_before.shape,
                                                                 p=0.5, n=1,
                                                                 dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    #全局(非lstm)参数, 这是对于嵌入行为和分类器而言的
    params = OrderedDict()
    #嵌入
    randn = np.rand(options['n_words'],
                    options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    #初始化分类器参数
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floadX)
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

    return params

#加载参数
def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)   #对权值进行了svd分解
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init lstm parameter
    初始化权值时,首先随机初始化权值,然后分解权值的奇异值,使用分解后的权值作为权值的初始化
    """
    #串联
    W = np.concatenate([ortho_weight(options['dim_proj']),
                         ortho_weight(options['dim_proj']),
                         ortho_weight(options['dim_proj']),
                         ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = np.concatenate([ortho_weight(options['dim_proj']),
                         ortho_weight(options['dim_proj']),
                         ortho_weight(options['dim_proj']),
                         ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = np.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None  #异常处理

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n+1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])  #For 2-D arrays it is equivalent to matrix multiplication
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.nnet.sigmoid(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)  # http://deeplearning.net/tutorial/lstm.html# (2)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                    tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

# 向前传播 (normal neural net),仅仅在分类之前,lstm之后有用
layers = {'lstm':(param_init_lstm, lstm_layer)}

def sgd(lr, tparams, grads, x, mask, y, cost):
    '''
    随机梯度下降算法(stochastic gradient descent)
     :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.
    :return:
    '''
    #新的共享变量集合将包含　a mini_batch的梯度
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]   #updates
    #计算mini-batch的梯度，但不需更新权重
    #function是一个由inputs计算outputs的对象，outputs一般是一个符号表达式，来表示如何计算从inputs->outputs
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')
    pup = [(p, p - lr * g) for p, g in zip(tparams.values(),gshared)]  ##?
    #根据前面计算好的梯度来更新权重
    f_update = theano.function([lr], [], updates=pup, name='sgd_fupdate')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, mask, y, cost):
    '''
    一个自适应的学习率优化器
    :param lr:      theano　sharedVariable
        initial learning rate
    :param tparams:     theano sharedVariable
        Model parameters
    :param grads: theano variable
        Gradients of cost w.r.t to parameres  *****
    :param x:   Theano variable
            Model input
    :param mask:    Theano variable
            Sequence mask
    :param y:Theano variable
            Targets
    :param cost:    Theano variable
        Objective function to minimize
    :return:

    Notes: reference arXiv:1212.5701.
    '''
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k)
                      for k, p in tparams.items()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup+rg2up, name='adadelta_f_grad_shared')
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for gz, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    rg2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    f_update = theano.function([lr], [], updates=rg2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    '''
    一个SGD变种，通过最近使用的步长的规范求平均值来处理step sizes: cales the step size by running average of the
    recent step norms.
    :param lr:
    :param tparams:
    :param grads:
    :param x:
    :param mask:
    :param y:
    :param cost:
    :return:
    '''
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    #Used for dropout
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    '''
    使用训练好的模型来计算新样本的概率
    :param f_pred_prob:
    :param prepare_data:
    :param data:
    :param iterator:
    :param verbose:
    :return:
    '''
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    '''
    计算 error
    :param f_pred: Theano fct computing the prediction
    :param prepare_data: 预备数据集
    :param iterator:
    :param verbose:
    :return:
    '''
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def train_lstm(
        dim_proj=128, #词嵌入的维数　和　LSTM隐藏层单元的数目
        patience=10, #Number of epoch to wait before early stop if no progress
        max_epochs=5000, # The maximum number of epoch to run
        dispFreq=10, #Display to stdout the training progress every N updates
        decay_c=0., #Weight decay(衰减)　for the classifier applied to the U weights
        lrate=0.0001, #随机梯度下降学习率(not used for adadelta and rmsprop)
        n_words=10000, #词典大小
        optimizer=adadelta, #可以用sgd,adadelta and rmsprop,sgd使用非常困难
        encoder='lstm', #使用lstm网络
        saveto='lstm_model.npz', #将最好的训练模型存储为lstm_model.npz
        validFreq=370, #更新370后验证错误率
        saveFreq=1110, #每隔1110次迭代保存一次参数
        maxlen=100, #最大序列值
        batch_size=16, #训练时处理数据的批大小
        valid_batch_size=64, #用于测试的批大小
        dataset='lmdb', #使用lmdb数据集

        #其他的一些参数
        noise_std=0, #噪声
        use_dropout=True, #使用dropout, if False slightly faster, but worst test error　
                            # # This frequently need a bigger model.
        reload_model=None, #保存模型数据的路径
        test_size=-1, #当test_size大于０，用来保存测试样本的数量
):

    #模型选择
    model_options = locals().copy() #return a dictionary containing the current scope's local variables.
    print("model options", model_options)
    #加载数据
    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    #加载训练，测试，评价数据集
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)  #用于模型评价的数据集为5%

    if test_size > 0:
        #由于测试集为根据数据大小排序好的数据，随机的选择测试集
        idx = np.arange(len(test[0]))
        np.random.shuffle(idx)  #shuffle操作
        idx = idx[:test_size]
        test = ([test[0][n] for n in range(idx)], [test[1][n] for n in idx])

    ydim = np.max(train[1])+1

    model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                            valid_err <= np.array(history_errs)[:,
                                         0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print('Train ', train_err, 'Valid ', valid_err,
                          'Test ', test_err)

                    if (len(history_errs) > patience and
                            valid_err >= np.array(history_errs)[:-patience,
                                         0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
    if saveto:
        np.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # train for all possible parameter and there definition.
    train_lstm(
        max_epochs=100, #最大数据块为100
        test_size=500, #测试数据为500
    )









