import odl
import numpy as np
from scipy.linalg import norm
import logging

from proxes.l2_data_fidelity import prox_l2_data_fidelity
from proxes.indicator_dual_unit_ball import prox_indicator_dual_unit_ball

logger = logging.getLogger(__name__)


def prox_TV(lambd, x, tol=1e-4, max_it=100, sigma=1/4, tau=1/4, gamma=0.2):
    """
    Approximates prox_{\lambda f}(x) for  f(x) = \|\nabla x\|_1 with PDHG
    :param x: current iterate
    :return: prox
    """

    # Setup gradient
    discr = odl.uniform_discr([0.]*len(x.shape), [*x.shape], x.shape)
    grad = odl.discr.diff_ops.Gradient(discr, pad_mode="order0")


    def operator(w):
        return grad(w).asarray()

    def adjoint(w):
        return grad.adjoint(w).asarray()

    def primal_prox(s, w):
        return prox_l2_data_fidelity(x, s, w)

    def dual_prox(t, w):
        return prox_indicator_dual_unit_ball(t, w, alpha=lambd)

    proxf = primal_prox
    proxg = dual_prox

    y = x
    ybar = x
    z = np.zeros(grad(x).asarray().shape)

    def get_cost(w):
        return 1 / 2 * norm((x - w).flatten(), ord=2) ** 2 + lambd * norm(grad(w).asarray().flatten(), ord=1)

    cost = get_cost(y)

    logger.info("Start solving prox TV | Initial cost = {}".format(cost))

    k = 1
    relerror = 1.

    while k <= max_it and relerror > tol:

        z1 = operator(ybar)
        z2 = z + tau * z1
        z3 = proxg(tau, z2)

        y1 = adjoint(z3)
        y2 = y - sigma * y1
        y3 = proxf(sigma, y2)

        theta = 1 / np.sqrt(1 + 2 * gamma * sigma)
        sigma = theta * sigma
        tau = tau / theta

        ybar = y3 + theta * (y3 - y)
        # calculate the norm of F first
        ynorm = norm((y - y3).flatten(), ord=2)
        znorm = norm((z - z3).flatten(), ord=2)
        nFx = np.sqrt(ynorm ** 2 + znorm ** 2)

        if k == 1:
            nFx0 = nFx

        y = y3
        z = z3

        cost = get_cost(y)
        relerror = nFx / (nFx0+1e-16)
        print(relerror)
        logger.info("Solving prox TV iteration {} | cost = {} | relative error = {}".format(k, cost, relerror))

        k += 1

    return y
