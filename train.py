#!/usr/bin/env python
# coding: utf-8

# ### Import

# In[1]:


import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import random
import math
import json
import functools
from dataclasses import dataclass, asdict
from collections import namedtuple

import pyteomics
from pyteomics import mgf, mass


# In[2]:


import tensorflow as tf
print(tf.__version__)

import tensorflow.keras as keras
import tensorflow.keras as k
from tensorflow.keras import backend as K
import tensorflow.experimental.numpy as tnp
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras import Model, Input
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy, MSE, MAE, cosine_similarity
from tensorflow.keras.losses import sparse_categorical_crossentropy

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.optimizers import RectifiedAdam as radam


# ### Help functions

# In[3]:


def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float32')

def np32(x): return np.array(x, dtype='float32')
def zero32(shape): return np.zeros(shape, dtype='float32')

def clipn(*kw, sigma=4):
    return np.clip(np.random.randn(*kw), -sigma, sigma) / sigma

class config(dict):
    def __init__(self, *args, **kwargs):
        super(config, self).__init__(*args, **kwargs)
        self.__dict__ = self

class data_seq(k.utils.Sequence):
    def __init__(self, sps, processor, batch_size, shuffle=1, xonly=1, **kws):
        self.sps = sps
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.xonly = xonly
        self.kws = kws

    def on_epoch_begin(self, ep):
        if ep > 0 and self.shuffle:
            np.random.shuffle(self.sps)

    def __len__(self):
        return math.ceil(len(self.sps) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sps))

        if self.xonly:
            return (self.processor(self.sps[start_idx: end_idx], **self.kws), )
        else:
            return self.processor(self.sps[start_idx: end_idx], **self.kws)


def fastmass(pep, ion_type, charge, nmod=None, mod=None, cam=True):        
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam: base += 57.021 * pep.count('C') / charge # fixed C modification
    
    return base

def m1(pep, c=1, **kws): return fastmass(pep, ion_type='M', charge=c, **kws)

def ppm(m1, m2):
    return ((m1 - m2) / m1) * 1000000

def ppmdiff(sp, pep=None):
    if pep is None: pep = sp['pep']
    mass = fastmass(pep, 'M', sp['charge'], mod=sp['mod'], nmod=sp['nmod'])
    return ((sp['mass'] - mass) / mass) * 1000000


# #### flat and vectorlize

# In[8]:

@nb.njit
def normalize(it, mode):
    if mode == 0:
        return it

    elif mode == 2: return np.sqrt(it)

    elif mode == 3: return np.sqrt(np.sqrt(it))

    elif mode == 4: return np.sqrt(np.sqrt(np.sqrt(it)))

    return it

@nb.njit
def remove_precursor(v, pre_mz, c, precision, low, r):
    for delta in (0, 1, 2):
        mz = pre_mz + delta / c
        if mz > 0 and mz >= low:
            pc = round((mz - low) / precision)

            if pc - r < len(v): v[max(0, pc - r): min(len(v), pc + r)] = 0
    return None # force inline

def kth(v, k):
    return np.partition(v, k)[k]


# In[10]:

@nb.njit
def mz2pos(mzs, pre, low): return round(mzs / pre + low)

@nb.njit
def flat(v, mz, it, pre, low, use_max):
    for i, x in enumerate(mz):
        pos = int(round((x - low) / pre))

        if pos < 0 or pos >= len(v): continue

        if use_max:
            v[pos] = max(v[pos], it[i])
        else:
            v[pos] += it[i]

    return v


# In[11]:


@nb.njit
def native_vectorlize(mz, it, mass, c, precision, dim, low, mode, v, kth, th, de, dn, use_max):
    it /= np.max(it)

    if dn > 0: it[it < dn] = 0

    it = normalize(it, mode) # pre-scale

    # if kth > 0: it = filterPeaks(it, _max_peaks=kth)

    flat(v, mz, it, precision, low, use_max)

    if de == 1: remove_precursor(v, mass, c, precision, low, r=1) #inplace, before scale

    v /= np.max(v) # final scale, de can change max

    return v

def vectorlize(mz, it, mass, c, precision, dim, low, mode, out=None, kth=-1, th=-1, de=1, dn=-1, use_max=0):
    if out is None: out = np.zeros(dim, dtype='float32')
    return native_vectorlize(asnp32(mz), np32(it), mass, c, precision, dim, low, mode, out, kth, th, de, dn, use_max)


# #### Process

# In[12]:


mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538, "Z": 147.0354, # oxidaed M
       }
mono = {k: v for k, v in sorted(mono.items(), key=lambda item: item[1])}

Alist = list('ACDEFGHIKLMNPQRSTVWYZ')
clist = ['*'] + Alist + [']', '[']
oh_dim = len(clist)

charMap = {aa: i for i, aa in enumerate(clist)}
idmap = {i: aa for i, aa in enumerate(clist)}
mlist = asnp32([0] + [mono[a] for a in Alist] + [0, 0])


# In[13]:

@nb.njit
def AA_pairs_native(seq, v):
    for i in range(len(seq) - 1):
        v[(seq[i] - 1) * 20 + seq[i + 1] - 1] = 1
    return v

def AA_pairs(seq, out=None):
    if out is None: out = np.zeros(400, dtype='float32')

    return AA_pairs_native(seq, out)

def compose(pep):
    cl = np.zeros(len(Alist), dtype='int32')
    for i, A in enumerate(Alist): cl[i] = pep.count(A)
    return cl

@nb.njit
def n_encode(seq, length, em):
    for i, aa in enumerate(seq):
        em[i][aa] = 1

    em[len(seq)][-1] = 1 # end char, no smooth

    em[len(seq) + 1:, 0] = 1 # padding, end + 1 to button
    return em

def encode(seq, length=-1, out=None):
    if length <= 0: length = len(seq)
        
    out = np.zeros((length, oh_dim), dtype='float32') if out is None else out

    return n_encode(seq, length, out)

def toseq(pep):
    return np.int32([charMap[c] for c in pep.upper()])


# In[15]:
#### load data

def spectra_ok(sp, ppm_threshold=10):
    mz, mass, pep, c = sp['mz'], sp['mass'], sp['pep'], sp['charge']

    if not pep.isalpha():
        return False # unknown mod

    if ppm_threshold > 0 and abs(ppmdiff(sp)) > ppm_threshold:
        return False

    return True

def filter_spectra(db):
    return [sp for sp in db if spectra_ok(sp)]


cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

def convert_mgf(sps):
    db = []

    for sp in sps:
        param = sp['params']

        if not 'charge' in param: raise
            
        c = int(str(param['charge'][0])[0])

        pep = title = param['title']
        if 'seq' in param: pep = param['seq']

        if 'pepmass' in param: mass = param['pepmass'][0]
        else: mass = float(param['parent'])

        rtime = 0 if not 'RTINSECONDS' in param else float(param['RTINSECONDS'])

        if 'hcd' in param:
            try:
                hcd = param['hcd']
                if hcd[-1] == '%':
                    hcd = float(hcd)
                elif hcd[-2:] == 'eV':
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass
                else:
                    raise Exception("Invalid type!")
            except:
                hcd = 0
        else: hcd = 0

        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'pep': pep, 'charge':c, 'mass': mass, 'mz': mz, 'it': it, 'nmod': 0,
                   'mod': np.zeros(len(pep), 'int32'), # currently no mod supported
                   'nce': hcd, 'title': title })

    return db

types = {'un': 0, 'cid': 1, 'etd': 2, 'hcd': 3, 'ethcd': 4, 'etcid': 5}

def readmgf(fn, type):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False, dtype='float32', use_index=False)

    codes = convert_mgf(data)
    
    for sp in codes:
        sp['type'] = types[type]
    return codes

def i2l(sps):
    sps = [sp.copy() for sp in sps]
    for sp in sps:
        sp['pep'] = sp['pep'].replace('I', 'L')
    return sps


# In[26]:
#### ResBlock

def norm_layer(norm):
    def get_norm_layer(**kws):
        if norm == 'bn':
            return BatchNormalization(**kws)
        if norm == 'bn0':
            normalizer = BatchNormalization({"gamma_initializer" :'zeros'}, **kws)
        elif norm == 'in':
            return InstanceNormalization(**kws)
        elif norm == 'ln':
            return LayerNormalization(**kws)
        elif norm is None or norm == 'none':
            return lambda x: x

        raise
    return get_norm_layer

def layerset(c2d=0, norm=None, ghost=0):
    if c2d:
        return config({
            'ConvLayer': k.layers.Conv2D,
            'UpSamplingLayer': k.layers.UpSampling2D,
            'MaxPoolingLayer': k.layers.MaxPooling2D,
            'AveragePoolingLayer': k.layers.AveragePooling2D,
            'GlobalPoolingLayer': k.layers.GlobalAveragePooling2D,
            'GlobalMaxLayer': k.layers.GlobalMaxPooling2D,
            'ZeroPadding': k.layers.ZeroPadding2D,
            'c2d': c2d,
            'normalizer': norm_layer(norm)
        })

    return config({
        'ConvLayer': k.layers.Conv1D,
        'UpSamplingLayer': k.layers.UpSampling1D,
        'MaxPoolingLayer': k.layers.MaxPooling1D,
        'AveragePoolingLayer': k.layers.AveragePooling1D,
        'GlobalPoolingLayer': k.layers.GlobalAveragePooling1D,
        'GlobalMaxLayer': k.layers.GlobalMaxPooling1D,
        'ZeroPadding': k.layers.ZeroPadding1D,
        'c2d': c2d,
        'normalizer': norm_layer(norm)
    })

def merge(o1, c1, c2d=0, strides=1, lset={}, norm=None, mact=None):
    lset = layerset(c2d, norm, **lset)

    layers = K.int_shape(c1)[-1]

    if strides > 1 or K.int_shape(o1)[-1] != layers:
        if strides > 1:
            o1 = lset.ZeroPadding((0, strides-1))(o1)
            o1 = lset.AveragePoolingLayer(strides)(o1)

        if K.int_shape(o1)[-1] != layers:
            o1 = lset.ConvLayer(layers, kernel_size=1, padding='same')(o1)

        o1 = lset.normalizer()(o1) # no gamma zero, main path

    if mact is None:
        return Add()([o1, c1])
    else:
        return Activation(mact)(Add()([o1, c1]))

def conv(x, layers, kernel, c2d=0, act='elu', lset={}, norm=None, dilation_rate=1,
        tcn=1, strides=1, se=0, **kws):
    lset = layerset(c2d, norm, **lset)

    if isinstance(kernel, int): kernel = (kernel,)
    for i, ks in enumerate(kernel):
        if i > 0: x = Activation(act)(x)

        x = lset.ConvLayer(layers, kernel_size=ks, padding='same',
                          strides=strides, dilation_rate=dilation_rate, **kws)(x)
        x = lset.normalizer()(x)

        for r in range(1, tcn):
            assert strides == 1 and dilation_rate == 1

            x = Activation(act)(x)
            x = lset.ConvLayer(layers, kernel_size=kernel,
                           padding='same', dilation_rate=2**r, **kws)(x)
            x = lset.normalizer()(x)

    return x

def res(x, l, ks, add=1, act='relu', c2d=False, norm=None, pool=2, strides=1,
        pooling='nil', lset={}, **kws):
    if not c2d: assert K.ndim(x) == 3
    else: assert K.ndim(x) == 4

    if pooling == 'up': x = lset.UpSamplingLayer(pool)(x)

    xc = conv(x, l, ks, c2d=c2d, lset=lset, norm=norm, act=act, strides=strides, **kws)

    if add: xc = merge(x, xc, c2d=c2d, lset=lset, norm=norm, mact=None, strides=strides)

    x = Activation(act)(xc) #final activation, xc to x naming

    if pooling == 1 or pooling == 'max': x = lset.MaxPoolingLayer(pool)(x)
    elif pooling == 2 or pooling == 'ave': x = lset.AvePoolingLayer(pool)(x)

    return x


# ### Denova start

# In[20]:


class hyper_para():
    @dataclass(frozen = True)
    class hyper():
        lmax: int = 30
        outlen: int = lmax + 2
        m1max: int = -1
        mz_max: int = 2048
        pre: float = 0.1
        low: float = 0
        dim: int = int(mz_max / pre)
        sp_dim: int = 4
        maxc: int = 8

        mode: int = 3
        kth: int = 50

        dynamic = config({'enhance': 1, 'bsz': 512})

        inputs = config({
            'y': ([sp_dim, dim], 'float32'),
            'info': ([2], 'float32'),
            'charge': ([maxc], 'float32')
        })

    def __init__(self):
        self.inner = self.__class__.hyper()

    def __getattr__(self, att):
        return getattr(self.inner, att)

    def dict(self):
        return asdict(self.inner)

class data_processor():
    def __init__(self, hyper):
        self.hyper = hyper

    # random drop peaks
    def data_enhance(self, mzs, its):
        its = normalize(its, self.hyper.mode)

        if len(mzs) > 80:
            th = kth(its, int(abs(clipn(sigma=2)) * self.hyper.kth))
            mzs = mzs[its > th] #mzs first
            its = its[its > th]
            
        return mzs, its

    def get_inputs(self, sps, training=1):
        hyper = self.hyper
        batch_size = len(sps)

        inputs = config({})
        for spec in hyper.inputs:
            inputs[spec] = np.zeros((batch_size, *hyper.inputs[spec][0]), dtype=hyper.inputs[spec][1])

        for i, sp in enumerate(sps):
            mass, c, mzs, its = sp['mass'], sp['charge'], sp['mz'], sp['it']
            mzs = mzs / 1.00052

            if training and hyper.dynamic.enhance:
                mzs, its = self.data_enhance(mzs, its)
            else:
                its = normalize(its, self.hyper.mode)

            inputs.info[i][0] = mass / hyper.mz_max
            inputs.info[i][1] = sp['type']
            inputs.charge[i][c - 1] = 1

            precursor_index = min(hyper.dim - 1, round((mass * c - c + 1) / hyper.pre))
            vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][0], use_max=1)
            inputs.y[i][1][:precursor_index] = inputs.y[i][0][:precursor_index][::-1] # reverse it

            vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][2], use_max=0)
            inputs.y[i][3][:precursor_index] = inputs.y[i][2][:precursor_index][::-1] # reverse mz

        return tuple([inputs[key] for key in inputs])

    def process(self, sps, training=1):
        hyper = self.hyper ##!

        batch_size = len(sps)

        rst = config({
            'peps': np.zeros((batch_size, hyper.outlen, oh_dim), dtype='float32')
        })

        mtl = config({
            'exist': ([len(Alist)], 'float32'),
            'nums': ([len(Alist)], 'float32'),
            'di': ([400], 'float32'),
            'length': ([hyper.outlen], 'float32'),
            'rk': ([1], 'float32'),
            'charge': ([1], 'int32'),
            'mass': ([1], 'float32')
        })

        for task in mtl:
            rst[task] = np.zeros((batch_size, *mtl[task][0]), dtype=mtl[task][1])

        for i, sp in enumerate(sps):
            pep, mass, c, mzs, its = sp['pep'], sp['mass'], sp['charge'], sp['mz'], sp['it']
            pep = sp['pep'].upper().replace('I', 'L')
            seq = toseq(pep)
            
            encode(seq, out=rst.peps[i])

            # aux tasks
            rst.mass[i] = mass / 4000
            rst.charge[i] = c - 1
            rst.length[i][len(pep)] = 1
            rst.rk[i] = int(pep[-1] == 'R' or pep[-1] == 'K')

            AA_pairs(seq, out=rst.di[i])
            rst.nums[i] = compose(pep)
            
            for c in pep:
                rst.exist[i][charMap[c] - 1] = 1

        inputs = self.get_inputs(sps, training=training)

        return (inputs, {key: rst[key] for key in rst})

hyper = hyper_para()
processor = data_processor(hyper)

# In[21]:
#### models

def bottomup(fu, norm='in', act='relu', **kws):
    v1 = fu[0]
    fu = fu[1:] # first is v1
    
    for u in fu:
        v1 = res(v1, K.int_shape(u)[-1], 5, act=act, strides=2, norm=norm, add=0, **kws)
        v1 = k.layers.Add()([v1, u])
#         v1 = Activation(act)(v1)

    return v1

class denovo_model():
    @staticmethod
    def sp_net(hyper, act='relu', norm='in'):
        inp = Input(shape=hyper.inputs['y'][0], name='sub_sp_inp')
        mz_inp = Input(shape=hyper.inputs['info'][0], name='sub_mz_inp')
        c_inp = Input(shape=hyper.inputs['charge'][0], name='sub_charge_inp')

        v1 = k.layers.Permute((2, 1))(inp)

        def sub_net(v1, act='relu'):
            for i, l in enumerate([8, 12]):
                v1 = res(v1, l, 7, norm=norm, add=1, act=act, strides=2)

            lst = []

            fils = np.int32([16, 24, 32, 48, 64]) * 12
            tcn = np.int32([8, 7, 6, 5, 4, ]) + 1

            for i, (l, r) in enumerate(zip(fils, tcn)):
                if i > 0:
                    v1 = res(v1, l, 9, norm=norm, add=1, act=act, strides=2)

                ext = r - 5
                if r > ext:
                    r = r - ext
                    ks = int(5 * 2 ** ext) - 1
                else:
                    ks = int(5 * 2 ** int(r - 1)) - 1
                    r = 1

                v1 = res(v1, l, ks, tcn=r, norm=norm, add=1, act=act)

                lst.append(v1)
            return v1, lst

        v1, lst = sub_net(v1)
        v1 = bottomup(lst[2:])

        v1 = k.layers.Permute((2, 1))(v1)
        v1 = res(v1, hyper.outlen, 1, act=act, norm=norm, add=0)
        v1 = k.layers.Permute((2, 1))(v1)

        l_size = K.int_shape(v1)[-2]
        infos = k.layers.Concatenate(axis=-1)([mz_inp, c_inp]) # meta infos
        infos = k.layers.Reshape((l_size, 1))(k.layers.Dense(l_size, activation='sigmoid')(infos))
        v1 = k.layers.Concatenate(axis=-1)([v1, infos])

        return k.models.Model([inp, mz_inp, c_inp, ], v1, name='sp_net')

    @staticmethod
    def auxiliary_tasks(v1, hyper, act='relu', norm='in'):        
        def vec_dense(x, nodes, name, act='sigmoid', layers=tuple(), **kws):
            for l in layers: x = res(x, l, 3, act='relu', **kws)
            x = k.layers.GlobalAveragePooling1D()(x)
        #     x = k.layers.Flatten()(x)
            x = k.layers.Dense(nodes, activation=act, name=name, dtype='float32')(x)
            return x

        aux_outputs = []

        aux_outputs.append(vec_dense(v1, 1, normal=norm, name='mass'))
        aux_outputs.append(vec_dense(v1, hyper.outlen, act='softmax', normal=norm, name='length'))
        aux_outputs.append(vec_dense(v1, 1, normal=norm, name='rk'))
        aux_outputs.append(vec_dense(v1, hyper.maxc, normal=norm, act='softmax', name='charge'))

        #aux exist:
        x = v1
        x = k.layers.GlobalAveragePooling1D()(x)
        x = k.layers.Dense(len(Alist))(x)
        x = Activation('sigmoid', name='exist', dtype='float32')(x)
        aux_outputs.append(x)

        #aux compose:
        x = v1
        x = k.layers.Permute((2, 1))(x)
        x = res(x, len(Alist), 1, act=act, norm=norm)
        x = k.layers.Permute((2, 1))(x)

        x = k.layers.Conv1D(hyper.lmax, kernel_size=1, padding='same')(x)
        x = k.layers.Activation('softmax', name='nums', dtype='float32')(x)
        aux_outputs.append(x)

        #aux AA pairs:
        x = v1
        x = k.layers.GlobalAveragePooling1D()(x)
        x = k.layers.Dense(400)(x)
        x = k.layers.Activation('sigmoid', name='di', dtype='float32')(x)
        aux_outputs.append(x) # don't merge

        return aux_outputs

    @staticmethod
    def build(hyper, act='relu', norm='in'):
        inp = Input(shape=hyper.inputs['y'][0], name='sp_inp')
        mz_inp = Input(shape=hyper.inputs['info'][0], name='mz_inp')
        c_inp = Input(shape=hyper.inputs['charge'][0], name='charge_inp')
        model_inputs = [inp, mz_inp, c_inp]

        spmodel = denovo_model.sp_net(hyper)
        sp_vector = spmodel(model_inputs)

        aux_outputs = denovo_model.auxiliary_tasks(sp_vector, hyper)
        
        final_pep = k.layers.Conv1D(oh_dim, kernel_size=1, padding='same', use_bias=1)(sp_vector)
        final_pep = k.layers.Activation('softmax', name='peps', dtype='float32')(final_pep)

        full_model = k.models.Model(inputs=model_inputs, outputs=aux_outputs + [final_pep], name='full_model')
        novo = k.models.Model(inputs=model_inputs, outputs=final_pep, name='denovo')

        return full_model, novo, spmodel


# In[22]:


class model_builder():
    class loss_fn:
        @staticmethod
        def mse_ce(fn=k.losses.categorical_crossentropy, c=0.25):
        #     @tf.function
            def mse(yt, yp):
                yt = K.cast(yt, yp.dtype)

                ce = fn(yt, yp)
                yt = K.cast(K.argmax(yt, axis=-1), 'float32') / 32.0
                yp = K.cast(K.argmax(yp, axis=-1), 'float32') / 32.0
                return c * k.losses.mean_absolute_error(yt, yp) + ce * 0.25
            return mse

        @staticmethod
        def mass_ce(fn=k.losses.categorical_crossentropy, c=0.001):
            def mse(yt, yp):
                yt = K.cast(yt, yp.dtype)
                ce_loss = fn(yt, yp)

                mp = K.sum(K.batch_flatten(yp * mlist), axis=-1)
                mt = K.sum(K.batch_flatten(yt * mlist), axis=-1)

                return c * k.losses.mean_absolute_error(mp, mt) + ce_loss
            return mse

        @staticmethod
        def mask_ce(yt, yp, ls=0.00):
            yts = K.argmax(yt, axis=-1)
            mask = K.cast(K.greater(yts, 0), dtype='int32')

            loss = k.losses.categorical_crossentropy(yt, yp, label_smoothing=ls)

            return K.sum(loss, axis=-1) / K.cast(K.sum(mask, axis=-1), dtype=K.floatx())

        @staticmethod
        def mask_acc(yt, yp):
            yts = K.argmax(yt, axis=-1)
            yps = K.argmax(yp, axis=-1)

            mask = K.cast(K.greater(yts, 0), dtype='int32')
            err = K.cast(K.not_equal(yts, yps), dtype='int32')
            return 1 - K.sum(err, axis=-1) / K.sum(mask, axis=-1)

        @staticmethod
        def full_acc(yt, yp):
            yts = K.argmax(yt, axis=-1)
            yps = K.argmax(yp, axis=-1)

            return K.cast(K.all(K.equal(yts, yps), axis=-1), dtype='float32')

    def __init__(self, hyper, denovo_model=denovo_model):
        self.hyper = hyper
        self.param = config()
        self.loss_fn = self.__class__.loss_fn
        self.denovo_model = denovo_model

    def set_loss(self):
        loss_fn = self.loss_fn

        self.losses = {
            "peps": loss_fn.mass_ce(fn=loss_fn.mask_ce, c=0.0001),
            "exist": binary_crossentropy,
            "nums": sparse_categorical_crossentropy,
            "di": binary_crossentropy,
            "mass": MSE,
            'rk': binary_crossentropy,
            "charge": sparse_categorical_crossentropy,
            'length': loss_fn.mse_ce(categorical_crossentropy)
        }

        self.weights = {
            "peps": 1,
            "nums": 0.2,
            "exist": 0.2,
            "di": 0.004, 'rk': 0.05,
            'mass': 0.1, "charge": 0.1, "length": 0.05
        }

        self.metrics = {
            "peps": [loss_fn.mask_acc, 'categorical_crossentropy',
                     'acc', loss_fn.mask_ce, loss_fn.full_acc],
            "nums": 'acc',
            "exist": 'binary_accuracy',
            'rk': 'acc',
            'di': 'binary_accuracy'
        }

    def build(self, summary=True):
        self.dm, self.novo, *self.rest = self.denovo_model.build(self.hyper)

        if summary: self.rest[0].summary()
        return self.dm, self.novo

    def compile(self, model=None, opt='adam'):
        self.set_loss()

        if model is None:
            self.dm.compile(optimizer=opt, loss=self.losses, loss_weights=self.weights, metrics=self.metrics)
        else:
            model.compile(optimizer=opt, loss=self.losses, loss_weights=self.weights, metrics=self.metrics)


# In[23]:
# #### start

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class train_mgr():
    def data_generator(self, sps, **kws):
        return data_seq(sps, processor.process, hyper.dynamic.bsz, xonly=0, **kws)

    def setup(self, **kws):
        self.his = tf.keras.callbacks.History()
        self.callbacks = [
            ModelCheckpoint('novo.hdf5', save_best_only=True, monitor='val_peps_mask_acc')
        ]

        self.builder = model_builder(hyper)
        self.dm, self.novo, *self.other_model = self.builder.build(**kws)
        return self.dm, self.novo

    def prepare_data(self):
        self.trainingset = i2l(filter_spectra(readmgf('train.mgf', 'hcd')))
        self.valset = i2l(filter_spectra(readmgf('validation.mgf', 'hcd')))

    def compile(self, bsz=None, lr=None):
        ### para
        hyper.dynamic.eps = 50
        hyper.dynamic.bsz = 32 * int(hyper.pre * 4 * 2.5) #* 2
        hyper.dynamic.lr = lr if lr else (hyper.dynamic.bsz / 1024) * 0.0009 * 16 * 6

        hyper.dynamic.opt = radam(lr=hyper.dynamic.lr)
        self.builder.compile(opt=hyper.dynamic.opt)
        return self.dm, self.novo

    def run(self):
        callbacks = self.callbacks + [self.his]

        self.dm.fit(self.data_generator(self.trainingset), epochs=hyper.dynamic.eps,
                    validation_data=self.data_generator(self.valset, training=0),
                    verbose=1, callbacks=callbacks)

# In[28]:

manager = train_mgr()

dm, novo = manager.setup(summary=1)

manager.compile()

manager.prepare_data()

manager.run()
