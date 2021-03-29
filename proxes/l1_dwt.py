import pywt

from proxes.l1_norm import prox_l1_norm


def prox_l1_dwt(lambd, x, wave="db4", level=2):
    v = pywt.wavedec2(x, wave, mode='zero', level=level)  # wavelet transform x
    for l in range(level+1):
        if l==0:
            v[0] = prox_l1_norm(lambd, v[0])
        else:
            v[l] = [prox_l1_norm(lambd, coeff) for coeff in v[l]]  # soft shrinkage in wavelet domain
    prox = pywt.waverec2(v, wave, mode='zero')  # transform wavelet prox back to image domain
    # periodization
    return prox

