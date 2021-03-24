import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pywt
import pywt.data

from gradients.l2_data_fidelity import gradient_l2_data_fidelity
from problems import Problem
from proxes.l1_norm import prox_l1_norm
from solvers.myula import MYULA

import pdb

data_var = 32
rng = np.random.default_rng()
wav = 'bior1.3'

#### Load cameraman image
x_offset = 60
y_offset = 100
size = 256

original = pywt.data.camera()[x_offset:x_offset+size,y_offset:y_offset+size]
noise = rng.normal(loc = 0, scale = data_var, size = original.shape)
data = original + noise

titles = ['Original image (x)', 'Noisy Image (y)']

fig = plt.figure(figsize=(6, 3))
for i, a in enumerate([original, original + noise]):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()


#### Wavelet transform of image, and plot approximation and details

coeffs = pywt.dwt2(original + noise, wav)
LL, (LH, HL, HH) = coeffs

cA_shape = LL.shape

top = np.concatenate([LL,LH], axis = 1)
bottom = np.concatenate([HL, HH], axis = 1)
data = np.concatenate([top, bottom], axis = 0)

titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

# coeffs = pywt.wavedec2(original + noise, wav, level = 2)
# data = coeffs[0].ravel()

# print(coeffs[0].shape)
# for i in range(len(coeffs)):
# 	for j in range(3):
# 		data = np.concatenate( (data, coeffs[i+1][j].ravel()) )


#### MYULA Steps

delta = 10
lambd = 2
n = 15000


def oper(x):
	#### Our forward operator is just the inverse wavelet transformation
	cA = x[:cA_shape[0],:cA_shape[1]]
	cH = x[:cA_shape[0],cA_shape[1]:]
	cV = x[cA_shape[0]:,:cA_shape[1]]
	cD = x[cA_shape[0]:,cA_shape[1]:]
	coeffs = (cA, (cH, cV, cD))
	# pdb.set_trace()
	return pywt.idwt2( coeffs, wavelet = wav )
	# return coeffs


def adj(x):
	#### Since the inverse wavelet transformation is orthogonal, its adjoint is the wavelet transformation
	LL, (LH, HL, HH) = pywt.dwt2(x, wav )

	top = np.concatenate([LL,LH], axis = 1)
	bottom = np.concatenate([HL, HH], axis = 1)
	data = np.concatenate([top, bottom], axis = 0)

	return data


def gradient(x):
	return gradient_l2_data_fidelity(x, data, var=data_var)


prob = Problem(shape=data.shape)


myula_mc = MYULA(problem = prob,
				 num_its = n,
				 burn_in = 100,
				 x_init = data,
				 gradient_f = gradient,
				 prox_g = prox_l1_norm,
				 delta = delta,
				 lambd = lambd
				 )

# print(myula_mc.compute_chain)

chain = myula_mc.compute_chain()

# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate(chain[:4]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(oper(a), interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title("Sample {}".format(i), fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()

im_chain = np.array([oper(a) for a in chain])
sample_mean = np.mean(im_chain, axis = 0)


titles = ['Original image (x)', 'Noisy Image (y)', "Posterior Mean"]
fig = plt.figure(figsize=(9, 3))
for i, a in enumerate([original, original + noise, sample_mean]):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()




