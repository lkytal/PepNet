# PredFull

__Visit [http://pepnet.com/](http://pepnet.com/) to try online prediction__

The first model for predicting complete tandem mass spectra from peptides sequences, using a deep CNN neural network trained on over 2 million experimental spectra.

## Update History

* 2021.12.28: First version.

## Method

Based on the structure of the residual convolutional networks. Current precision (bin size): 0.1 Th.

![model](imgs/model.png)

## How to use

__Expect clone this project, you should download `model.h5` from [zenodo.org](https://zenodo.org/record/5807120) and place it into this folder.__

### Important Notes

* Will only output unmodification sequences.
* This model assumes a __FIXED__ carbamidomethyl on C
* The length of output peptides are limited to =< 30

### Required Packages

Recommend to install dependency via [Anaconda](https://www.anaconda.com/distribution/)

* Python >= 3.7
* Tensorflow >= 2.5.0
* Pandas >= 0.20
* pyteomics

### Output format

Sample output looks like:

TITLE | DENOVO | Score | PPM Difference | Positional Score
------- | ------ | ---- | ------- | ------ 
spectra 1 | LALYCHQLNLCSK | 1.0000 | -3.8919184 | [1.0, 0.9999956, 1.0, 1.0, 1.0, 1.0, 0.99999976, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
spectra 2 | HEELMLGDPCLK | 1.0000 | 4.207922 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99999976, 1.0]
spectra  3 | AGLVGPEFHEK | 1.0000 | 0.54602236 | [1.0, 1.0, 1.0, 1.0, 1.0, 0.99999917, 1.0, 1.0, 1.0, 1.0, 1.0]

### Usage

Simply run:

`python denovo.py --input example.mgf --model model.h5 --output example_prediction.tsv`

The output file is in MGF format

* --input: the input mgf file
* --output: the output file path
* --model: the pretrained model

## Prediction Examples

We provide sample data on [zenodo](https://zenodo.org/record/5807120) for you to evaluate the sequencing performance. The `example.mgf` file on google drive contains ground truth spectra (randomly sampled from [NIST Human Synthetic Peptide Spectral Library](https://chemdata.nist.gov/dokuwiki/doku.php?id=peptidew:lib:kustersynselected20170530)), while the `example.tsv` file contains pre-run predictions.

## How to build & train the model

For those who are interested in reproducing this model, here we provide `train_model.py` of example codes to build and train the model.

