def prox_l2_norm(lambd, x, var=None):
    """
    Computes prox_{\lambda f}(x) for  f(x) = \frac{1}{2}\|x\|_2^2, which is given by \frac{1}{1 + \lambda} x
    :param x: current iterate
    :return: prox \frac{1}{2}x
    """
    # TODO variance must be matrix
    # TODO if var is None assume that we have unit variance
    return 1/(1+lambd) * x