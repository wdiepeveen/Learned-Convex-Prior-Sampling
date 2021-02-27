import numpy as np
import logging

logger = logging.getLogger(__name__)


class MYULA:
    def __init__(self,
                 problem=None,
                 num_its=1000,
                 burn_in=1000,
                 thinning=1,
                 x_init=None,
                 gradient_f=None,
                 prox_g=None,
                 delta=None,
                 lambd=None,
                 ma=True,
                 seed=0
    ):

        self.problem = problem

        self.num_its = num_its
        self.burn_in = burn_in
        self.thinning = thinning

        self.gradient = gradient_f
        self.prox = prox_g

        self.delta = delta
        self.lamda = lambd

        self.ma = ma

        self.seed = seed
        np.random.seed(seed)

        if x_init is None:
            self.x = np.zeros(self.problem.shape)
        else:
            self.x = x_init

    def compute_chain(self):
        logger.info("Start burn-in steps")
        for i in range(self.burn_in):
            self.do_step()

        logger.info("Finished burn-in steps")

        num_slots = int(np.floor(self.num_its/self.thinning))
        X = np.zeros((num_slots,) + self.problem.shape)
        k = 0
        logger.info("Start Markov chain steps")
        for i in range(self.num_its):
            self.do_step()
            if self.num_its % self.thinning == 0:
                logger.info("Iteration {}/{}".format(i+1, self.num_its))
                X[k] = self.x
                k+=1

        return X

    def do_step(self):
        z = np.random.randn(*self.problem.shape)
        x = self.x - self.delta * self.gradient(self.x) \
            - self.delta/self.lamda * (self.x - self.prox(self.lamda, self.x)) \
            + np.sqrt(2*self.delta) * z

        # TODO implement Metropolis Adjustment step

        self.x = x

