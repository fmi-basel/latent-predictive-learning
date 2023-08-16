#!/usr/bin/env python

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import *
from scipy.io import mmread

import argparse

parser = argparse.ArgumentParser(description='Plot weight histogram.')
parser.add_argument('--input', type=str, default='lastrun/input_con.0.wmat', help='input file')
parser.add_argument('--output', type=str, default="weight_hist.pdf", help='output file')
args = parser.parse_args()

mat = mmread(args.input).todense()
hist = mat.mean(axis=1)

fig = plt.figure(figsize=(3.3,2))
plt.subplot(2,1,1)
for i in range(10):
    dat=mat[:,i]
    idx=np.where(dat)[0]
    val=np.squeeze([ dat[k] for k in idx ])
    plt.plot(idx, val)
plt.ylabel("Weight")

a=100
n=5
means = []
stdevs = []
for i in range(n):
    means.append(hist[i*a:(i+1)*a].mean())
    stdevs.append(hist[i*a:(i+1)*a].std())

plt.subplot(2,1,2)
plt.bar(np.arange(n), means, yerr=stdevs)
plt.ylabel("Weight")
plt.tight_layout()

sns.despine()

plt.savefig(args.output)

# plt.show()
