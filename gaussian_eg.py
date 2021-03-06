import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gradients.l2_data_fidelity import gradient_l2_data_fidelity
from problems import Problem
from proxes.l2_norm import prox_l2_norm
from solvers.myula import MYULA

data = np.ones(2)
data_var = 0.5

delta = 0.1
lambd = 0.2
n = 5000

def gradient(x):
	return gradient_l2_data_fidelity(x,data,var=data_var)

prob = Problem(shape=data.shape)

myula_mc = MYULA(problem = prob,
				 num_its = n,
				 burn_in = 1000,
				 x_init = data,
				 gradient_f = gradient,
				 prox_g = prox_l2_norm,
				 delta = delta,
				 lambd = lambd
				 )

# print(myula_mc.compute_chain)

chain = myula_mc.compute_chain()

mean = np.mean(chain, axis = 0)
var = np.var(chain, axis = 0)

print(mean)
print(var)

# print(chain.T)
# plt.hist2d(chain.T[0], chain.T[1], bins= 20, density = True)
# plt.show()

sns.jointplot(x=chain.T[0], y=chain.T[1], kind="kde")
plt.show()