def gradient_l2_data_fidelity(x, data, var=1, operator=None, adjoint=None):
    """
    Computes gradient of \frac{1}{2var}\|Ax - y\|_2^2, which is given by 1/var * A^*(Ax-f) for sigma^2=var
    :param x: current iterate
    :param data: data term y
    :param operator: forward operator A. If None this is the identity
    :param adjoint: adjoint of forward operator A. Must be provided if operator is provided
    :return: gradient 1/var * A^*(Ax-f)
    """
    # TODO variance should be a matrix
    if operator is not None:
        if adjoint is not None:
            tmp = operator(x)
            gradient = 1/var * adjoint(tmp - data)
        else:
            raise RuntimeError("operator given without adjoint")

    else:
        gradient = 1/var * (x - data)

    return gradient