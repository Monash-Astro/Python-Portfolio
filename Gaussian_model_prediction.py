import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.timeseries import LombScargle
import george
from george import kernels
import scipy.optimize as op
import emcee

"""
Use gaussian processes to predict the brightness of a star for the next two months.
"""

with open("star_data.pkl", "rb") as fp:
    data = pickle.load(fp)

t, y, yerr = data['t'], data['y'], data['yerr']

# Plot initial data
plt.figure()
plt.errorbar(t, y, yerr)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Stellar data')
plt.show()
plt.savefig('Stellar data.png')

frequency, power = LombScargle(t, y, yerr).autopower()

plt.figure()
plt.plot(frequency, power)
plt.xlabel('frequency')
plt.ylabel('power')
plt.title('Best fit frequencies')
plt.show()
plt.title('Best fit frequencies')

i1 = np.argmax(power[:int(len(power)/3)])
i2 = int(len(power)/3)+np.argmax(power[int(len(power)/3): int(2*len(power)/3)])
i3 = int(2*len(power)/3)+np.argmax(power[int(2*len(power)/3):])-1
i_max = [i1,i2,i3]
f_max = [frequency[i] for i in i_max]
ls = LombScargle(t, y, yerr)
y_fits = [ls.model(t, f) for f in f_max]

plt.figure()
plt.errorbar(t, y, yerr)
for i in range(3):
#   plt.plot(t, y_fits[i])
  plt.plot(np.linspace(min(t),max(t), 10000), 
            (1/(i+1))*np.sin(2 * np.pi * f_max[i] * np.linspace(min(t),max(t), 10000)), 
            label=f'p = ' + str(1/f_max[i]))
plt.xlabel('time')
plt.ylabel('y')
plt.title('Best fit frequencies')
plt.legend()
plt.show()
plt.savefig('Best fit frequencies.png')

p0, p1, p2 = 1/f_max[0], 1/f_max[1], 1/f_max[2]

k1 = kernels.Matern32Kernel(1.0)
k2 = kernels.ExpSine2Kernel(gamma=1.0, log_period=np.log(p0))
k3 = kernels.RationalQuadraticKernel(log_alpha=1, metric=1)     # Optional to set kernal 3 as well (takes significantly longer)
kernel = k1 + k2 + k3  #uncomment kernel 3 if not wanted

gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(0.19**2), fit_white_noise=True)

# Compute the GP once before starting the optimization.
gp.compute(t)
# print(gp.log_likelihood(y))
# print(gp.grad_log_likelihood(y))

#######################  construct kernels and optimise the hyperparameters  ####################### 

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

print('initial ln-likelihood: ',gp.log_likelihood(y))
#   optimization
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
gp.set_parameter_vector(results.x)
print('final log-likelihood: ',gp.log_likelihood(y))

x = np.linspace(max(t), max(t)+61, 1000)
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)
        
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(t, y, s=1, c="k")
plt.plot(x, mu, c='purple', label='mean path')
ax.fill_between(x, mu+std, mu-std, color="pink", alpha=0.5)

        
# ax.set_xlim(t.min(), 2025)
ax.set_xlabel(r"year")
ax.set_ylabel(r"Brightness")
plt.legend()
ax.set_title('Mean predicted path (w. 2 SD shaded)')
fig.tight_layout()
plt.savefig('Mean predicted path (w. 2 SD shaded).png')

#######################  sample parameters and make predictions  ####################### 

def lnprob(p):
    # Trivial uniform prior.
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.lnlikelihood(y, quiet=True)
    
gp.compute(t)

# Set up the sampler.
nwalkers, ndim = len(gp)*2, len(gp)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

# Initialize the walkers.
w0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)

print("Running burn-in")
w0, _, _ = sampler.run_mcmc(w0, 10)

print("Running production chain")
sampler.run_mcmc(w0, 60);

fig, ax = plt.subplots(figsize=(6, 6))

x = np.linspace(max(t), max(t)+61, 1000)
for i in range(50):
    # Choose a random walker and step.
    w = np.random.randint(sampler.chain.shape[0])
    n = np.random.randint(sampler.chain.shape[1])
    gp.set_parameter_vector(sampler.chain[w, n])

    # Plot a single sample.
    ax.plot(x, gp.sample_conditional(y, x), "purple", alpha=0.1)

ax.scatter(t, y, c="k", s=1)

ax.set_xlim(t.min(), max(t)+61)
ax.set_xlabel(r"year")
ax.set_ylabel(r"Brightness")
ax.set_title('Sample draws - predicted brightness')
fig.tight_layout()
plt.savefig('Sample draws - predicted brightness.png')
