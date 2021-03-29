import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import emcee as mc

from gradients.l2_data_fidelity import gradient_l2_data_fidelity
from problems import Problem
from proxes.l2_norm import prox_l2_norm
from solvers.myula import MYULA

data = np.ones(2)
data_var = 0.5

delta = 0.1
lambd = 0.2
n = 6000

def gradient(x):
	return gradient_l2_data_fidelity(x,data,var=data_var)

def nlog_likelihood(x, y, data_var = 1):
	# \frac{1}{2var}\|Ax - y\|_2^2
	return ( np.linalg.norm( x - y )**2 ) / (2.0 * data_var)

def nlog_prior(x):
	return ( np.linalg.norm(x)**2 ) / 2.0


prob = Problem(shape=data.shape)

myula_mc = MYULA(problem = prob,
				 num_its = n,
				 burn_in = 1000,
				 thinning = 1,
				 x_init = data,
				 gradient_f = gradient,
				 prox_g = prox_l2_norm,
				 delta = delta,
				 lambd = lambd
				 )

# print(myula_mc.compute_chain)


chain = myula_mc.compute_chain()[::]
iterative_means = np.zeros(chain.shape)
for i in range(len(chain)):
	iterative_means[i] = np.mean(chain[:i], axis = 0)
iterative_norms = np.zeros(chain.shape[0] - 1)
for i in range(len(chain) - 1):
	iterative_norms[i] = np.linalg.norm(iterative_means[i+1] - iterative_means[i])


print(chain.shape)

x_chain = chain[:,0]
y_chain = chain[:,1]

x_auto_correlation = mc.autocorr.function_1d(x_chain)
y_auto_correlation = mc.autocorr.function_1d(y_chain)

plt.plot(x_auto_correlation[:30], "b-", label="First Coordinate")
plt.plot(y_auto_correlation[:30], "r-", label="Second Coordinate")
plt.title("Autocorrelation Function")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.legend(loc = "upper right")
plt.show()

print("X autocorrelation time: ", mc.autocorr.integrated_time(x_chain))
print("Y autocorrelation time: ", mc.autocorr.integrated_time(y_chain))


nlog_liks_samples = np.array([nlog_likelihood(a, data, data_var) + nlog_prior(a) for a in chain])
nlog_liks_means = np.array([nlog_likelihood(a, data, data_var) + nlog_prior(a) for a in iterative_means])
plt.semilogy(nlog_liks_samples, "r-", label="Sample likelihoods")
plt.semilogy(nlog_liks_means, "b-", label="Likelihood of iterative means")
plt.xlabel("Number of samples")
plt.ylabel("Negative log likelihoods")
plt.legend(loc = "upper right")
plt.show()


mean = np.mean(chain, axis = 0)
var = np.var(chain, axis = 0)

print("Unthinned Mean: ", mean)
print("Unthinned Var: ", var)

print("")
print("Thinned Mean: ", np.mean(chain[::6,:], axis = 0))
print("Thinned Var: ", np.var(chain[::6,:], axis = 0))


plt.semilogy(iterative_norms)
plt.xlabel("Iterations")
plt.ylabel("Successive Iterative Distance")
plt.show()


# print(chain.T)
# plt.hist2d(chain.T[0], chain.T[1], bins= 20, density = True)
# plt.show()

# sns.jointplot(x=x_chain[::6], y=y_chain[::6], kind="kde")
# plt.show()