import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pywt
import pywt.data

from gradients.l2_data_fidelity import gradient_l2_data_fidelity
from problems import Problem
from proxes.l1_norm import prox_l1_norm
from proxes.TV import prox_TV
from solvers.myula import MYULA

import pdb

data_var = 24
rng = np.random.default_rng()
wav = 'bior1.3'
levs = None

#### Load cameraman image
x_offset = 60
y_offset = 100
size = 256

original = pywt.data.camera()[x_offset:x_offset+size,y_offset:y_offset+size]
noise = rng.normal(loc = 0, scale = data_var, size = original.shape)
# data = original + noise

# titles = ['Original image (x)', 'Noisy Image (y)']

# fig = plt.figure(figsize=(6, 3))
# for i, a in enumerate([original, original + noise]):
#     ax = fig.add_subplot(1, 2, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()


#### Wavelet transform of image, and plot approximation and details

coeffs = pywt.wavedec2(original + noise, wav, level = levs)
data, slices = pywt.coeffs_to_array(coeffs)

# LL, (LH, HL, HH) = coeffs

# type(data)

# plt.imshow( data, cmap=plt.cm.gray )
# plt.show()

# im_rec = pywt.waverec2(pywt.array_to_coeffs(data, slices, 'wavedec2'), wav)
# plt.imshow( im_rec, cmap = plt.cm.gray )
# plt.show()


# titles = ['Approximation', ' Horizontal detail',
#           'Vertical detail', 'Diagonal detail']

# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()

# coeffs = pywt.wavedec2(original + noise, wav, level = 2)
# data = coeffs[0].ravel()

# print(coeffs[0].shape)
# for i in range(len(coeffs)):
# 	for j in range(3):
# 		data = np.concatenate( (data, coeffs[i+1][j].ravel()) )

titles = ['Original image (x)', 'Noisy Image (y)', 'Wavelet transformation (Dy)']

fig = plt.figure(figsize=(9, 3))
for i, a in enumerate([original, original + noise, data]):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()



#### MYULA Steps

delta = 0.2
lambd = 1
n = 4000


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
				 burn_in = 10,
				 thinning = 1,
				 x_init = data,
				 gradient_f = gradient,
				 prox_g = prox_l1_norm,
				 delta = delta,
				 lambd = lambd
				 )

# print(myula_mc.compute_chain)

chain = myula_mc.compute_chain(print_logs = True)

# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate(chain[:4]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(oper(a), interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title("Sample {}".format(i), fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()

# im_chain = np.array([a for a in chain])
im_chain = np.array([pywt.waverec2(pywt.array_to_coeffs(a, slices, 'wavedec2'), wav) for a in chain])

sample_mean = np.mean(im_chain, axis = 0)
sample_var = np.var(im_chain, axis = 0)

# titles = ['Original image (x)', 'Noisy Image (y)', "Posterior Mean", "Pixelwise Variance"]
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([original, original + noise, sample_mean, sample_var]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()

# titles = ["Posterior Mean", "Pixelwise Variance"]
# fig = plt.figure(figsize=(6, 3))
# for i, a in enumerate([sample_mean, sample_var]):
#     ax = fig.add_subplot(1, 2, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()

# titles = ["Posterior Mean", "Pixelwise Variance"]
fig = plt.figure(figsize=(15, 3))
for i, a in enumerate(im_chain[1000:1005]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title("Sample {}".format(i), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()



