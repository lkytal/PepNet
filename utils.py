import argparse
import numpy as np
import numba as nb
from numba import int32, float32, float64, boolean
import math
from pyteomics import mgf, mass

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras as k

class config(dict):
    def __init__(self, *args, **kwargs):
        super(config, self).__init__(*args, **kwargs)
        self.__dict__ = self

def f4(x): return "{0:.4f}".format(x)

def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float32')
def np32(x): return np.array(x, dtype='float32')
def clipn(*kw, sigma=4):
    return np.clip(np.random.randn(*kw), -sigma, sigma) / sigma


def fastmass(pep, ion_type, charge, mod=None, cam=True):
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam:
        base += 57.021 * pep.count('C') / charge

    if not mod is None:
        base += 15.995 * np.sum(mod == 1) / charge
    return base


class data_seq(k.utils.Sequence):
    def __init__(self, sps, processor, batch_size, shuffle=1, xonly=1):
        self.sps = sps
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.xonly = xonly

    def on_epoch_begin(self, ep):
        if ep > 0 and self.shuffle:
            np.random.shuffle(self.sps)

    def __len__(self):
        return math.ceil(len(self.sps) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sps))

        if self.xonly:
            return (self.processor(self.sps[start_idx: end_idx]), )
        else:
            return self.processor(self.sps[start_idx: end_idx])

def m1(pep, c=1, **kws): return fastmass(pep, ion_type='M', charge=c, **kws)

def ppmdiff(sp, pep):
    mass = fastmass(pep, 'M', sp['charge'])
    return ((sp['mass'] - mass) / mass) * 1000000

def ppm(m1, m2):
    return ((m1 - m2) / m1) * 1000000

mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538, "Z": 147.0354,  # oxidaed M
        }
mono = {k: v for k, v in sorted(mono.items(), key=lambda item: item[1])}

amino_list = list('ACDEFGHIKLMNPQRSTVWYZ')
oh_dim = len(amino_list) + 3 #one_hot dimension

amino2id = {'*': 0, ']': len(amino_list) + 1, '[': len(amino_list) + 2}
for i, a in enumerate(amino_list):
    amino2id[a] = i + 1

id2amino = {0: '*', len(amino_list) + 1: ']', len(amino_list) + 2: '['}
for a in amino_list:
    id2amino[amino2id[a]] = a

mass_list = asnp32([0] + [mono[a] for a in amino_list] + [0, 0])

@nb.njit
def normalize(it, mode):
    if mode == 0:
        return it
    elif mode == 2: return np.sqrt(it)

    elif mode == 3: return np.sqrt(np.sqrt(it))

    elif mode == 4: return np.sqrt(np.sqrt(np.sqrt(it)))

    return it

@nb.njit
def _remove_precursor(v, pre_mz, c, precision, low, r):
    for delta in (0, 1, 2):
        mz = pre_mz + delta / c
        if mz > 0 and mz >= low:
            pc = round((mz - low) / precision)

            if pc - r < len(v):
                v[max(0, pc - r): min(len(v), pc + r)] = 0

    return None # force inline

def remove_precursor(v, pre_mz, c, precision, low, r=1):
    return _remove_precursor(v, pre_mz, c, precision, low, r)

@nb.njit
def filterPeaks(v, _max_peaks):
    if _max_peaks <= 0 or len(v) <= _max_peaks: return v

    kth = len(v) - _max_peaks
    peak_thres = np.partition(v, kth)[kth]
    v[v < peak_thres] = 0
    return v


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

@nb.njit
def _vectorlize(mz, it, mass, c, precision, dim, low, mode, v, kth, th, de, dn, use_max):
    it /= np.max(it)

    if dn > 0: it[it < dn] = 0

    it = normalize(it, mode) # pre-scale

    if kth > 0: it = filterPeaks(it, _max_peaks=kth)

    flat(v, mz, it, precision, low, use_max)

    if de == 1: _remove_precursor(v, mass, c, precision, low, r=1) #inplace, before scale

    v /= np.max(v) # final scale, de can change max

    return v

def vectorlize(mz, it, mass, c, precision, dim, low, mode, out=None, kth=-1, th=-1, de=1, dn=-1, use_max=0):
    if out is None: out = np.zeros(dim, dtype='float32')
    return _vectorlize(asnp32(mz), np32(it), mass, c, precision, dim, low, mode, out, kth, th, de, dn, use_max)


def decode(seq2d):
    return np.int32([np.argmax(seq2d[i]) for i in range(len(seq2d))])

def topep(seq):
    return ''.join(map(lambda n: id2amino[n], seq)).strip("*[]")

def toseq(pep):
    return np.int32([amino2id[c] for c in pep.upper()])

def what(seq2d):
    return topep(decode(seq2d))

def clean(pep):
    return pep.strip("*[]").replace('I', 'L').replace('*', 'L').replace('[', 'A').replace(']', 'R')


def iterate(x, bsz):
    while len(x) > bsz:
        yield x[:bsz]
        x = x[bsz:]
    yield x