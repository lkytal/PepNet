import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
from pyteomics import mgf, mass
from dataclasses import dataclass, asdict

import tensorflow as tf
import tensorflow.keras as k
from tensorflow_addons.layers import InstanceNormalization

from utils import *


print(tf.__version__, tf.config.list_physical_devices('GPU'))


def read_mgf(data, count=-1, default_charge=-1):
    collision_const = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
    spectra = []

    for sp in data:
        param = sp['params']

        if not 'charge' in param:
            if default_charge != -1:
                c = default_charge
            else:
                raise AttributeError("MGF contains spectra without charge")
        else:
            c = int(str(param['charge'][0])[0])

        if 'seq' in param:
            pep = param['seq'].strip()
        elif 'title' in param:
            pep = param['title'].strip()
        else:
            pep = ''

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        try:
            hcd = param['hcd']
            if hcd[-1] == '%':
                hcd = float(hcd)
            elif hcd[-2:] == 'eV':
                hcd = float(hcd[:-2])
                hcd = hcd * 500 * collision_const[c] / mass
            else:
                raise Exception("Invalid eV format!")
        except:
            hcd = 0

        mz = sp['m/z array']
        it = sp['intensity array']

        spectra.append({'pep': pep, 'charge': c, 'type': 3, 'nmod': 0, 'mod': np.zeros(len(pep), 'int32'),
                    'mass': mass, 'mz': mz, 'it': it, 'nce': hcd})

        if count > 0 and len(spectra) >= count:
            break

    return spectra

# post correction step
def post_correction(matrix, mass, c, ppm=10):
    positional_score = np.max(matrix, axis=-1)
    seq = decode(matrix)
    pep = topep(seq)
    seq = seq[:len(pep)]
    tol = mass * ppm / 1000000

    for i, char in enumerate(pep):
        if char in '*[]':
            pep = pep[:i]
            positional_score[i:] = 1
            seq = seq[:i]
            break

    if len(pep) < 1:
        return '', -1, positional_score

    msp = m1(topep(seq), c)
    delta = msp - mass
    pos = 0
    a = seq[0]

    if abs(delta) < tol:
        return topep(seq), -1, positional_score

    for i in range(len(seq) - 1):  # no last pos
        mi = mass_list[seq[i]]
        for j in range(1, 21):
            if j == 8:
                continue  # ignore 'I'

            d = msp - mass + (mass_list[j] - mi) / c

            if abs(d) < abs(delta):
                delta = d
                pos = i
                a = j

    if abs(delta) < tol:  # have good match
        candi = np.int32(seq == seq[pos])
        if np.sum(candi) > 1.5:  # ambiguis
            pos = np.argmin((1 - candi) * 10 + candi *
                            np.max(matrix[:len(seq)], axis=-1))

        seq[pos] = a
        positional_score[pos] = 1

        return topep(seq), pos, positional_score
    else:
        return topep(seq), -1, positional_score

# hyper parameter
@dataclass(frozen = True)
class hyper():
    lmax: int = 30
    outlen: int = lmax + 2
    m1max: int = 2048
    mz_max: int = 2048
    pre: float = 0.1
    low: float = 0
    vdim: int = int(mz_max / pre)
    dim: int = vdim + 0
    maxc: int = 8
    sp_dim: int = 4

    mode: int = 3
    scale: float = 0.3

# convert spectra into model input format
def input_processor(spectra):
    nums = len(spectra)

    inputs = config({
        'y': np.zeros([nums, hyper.sp_dim, hyper.dim], 'float32'),
        'info': np.zeros([nums, 2], 'float32'),
        'charge': np.zeros([nums, hyper.maxc], 'float32')
    })

    for i, sp in enumerate(spectra):
        mass, c, mzs, its = sp['mass'], sp['charge'], sp['mz'], sp['it']
        mzs = mzs / 1.00052

        its = normalize(its, hyper.mode)

        inputs.info[i][0] = mass / hyper.m1max
        inputs.info[i][1] = sp['type']
        inputs.charge[i][c - 1] = 1

        precursor_index = min(hyper.dim - 1, round((mass * c - c + 1) / hyper.pre))

        vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][0], use_max=1)
        inputs.y[i][1][:precursor_index] = inputs.y[i][0][:precursor_index][::-1] # reverse it

        vectorlize(mzs, its, mass, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][2], use_max=0)
        inputs.y[i][3][:precursor_index] = inputs.y[i][2][:precursor_index][::-1] # reverse mz

    return tuple([inputs[key] for key in inputs])

def denovo(model, spectra, batch_size):
    predict_peps = []
    scores = []
    positional_scores = []
    charges = [sp['charge'] for sp in spectra]
    peps = [sp['pep'] for sp in spectra]

    predictions = model.predict(data_seq(spectra, input_processor, batch_size, xonly=True), verbose=1)

    for rst, sp in zip(predictions, spectra):
        ms, c = sp['mass'], sp['charge']

        # run post correction
        pep, pos, positional_score = post_correction(rst, ms, c)

        predict_peps.append(pep)
        positional_scores.append(positional_score)
        scores.append(np.prod(positional_score))

    ppm_diffs = asnp32([ppm(sp['mass'], m1(pp, c)) for sp, pp, c in zip(spectra, predict_peps, charges)])
    return peps, predict_peps, scores, positional_scores, ppm_diffs, spectra


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='input file path', default='example.mgf')
parser.add_argument('--output', type=str,
                    help='output file path', default='example.tsv')
parser.add_argument('--model', type=str,
                    help='Pretained model path', default='model.h5')
parser.add_argument('--loop_size', type=int,
                    help='number of spectra in memory', default=10000)
parser.add_argument('--default_charge', type=int,
                    help='default charge for spectra without charges, -1 means disabled', default=-1)
parser.add_argument('--batch_size', type=int,
                    help='number of spectra per step', default=128)

args = parser.parse_args()

print('Loading model....')
tf.keras.backend.clear_session()
model = k.models.load_model(args.model, compile=0)

print("Starting reading mgf of:", args.input)
input_stream = mgf.read(open(args.input, "r"), convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)

f = open(args.output, 'w+')
f.writelines(['TITLE\tDENOVO\tScore\tPPM Difference\tPositional Score\n'])

# sequencing loop
i = 0
while True:
    spectra = read_mgf(input_stream, count=args.loop_size, default_charge=args.default_charge)
    if len(spectra) <= 0:
        break

    print("De novo spectra from", i, "to", i + len(spectra))
    i += len(spectra)

    peps, ppeps, scores, pscores, ppms, _ = denovo(model, spectra, args.batch_size)

    f.writelines("\t".join([p, pp, f4(s), str(ppm), str(list(pscore)[:len(pp)])]) + "\n"
                 for p, pp, s, pscore, ppm in zip(peps, ppeps, scores, pscores, ppms))

f.close()
print('Finished,', i, 'spectra in total')
