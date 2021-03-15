def prox_l2_data_fidelity(f, lambd, x, var=None):

    assert f is not None
    # TODO variance must be matrix
    # TODO if var is None assume that we have unit variance
    return 1/(1+lambd) * (x + lambd * f)