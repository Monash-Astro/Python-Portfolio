# import necessary modules and functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmdstanpy as stan
import seaborn as sns
from scipy.stats import multivariate_normal
# from scipy import random

"""
Stan Challenge:
A company has found that the widgets they manufacture contain some mysterious radioactive
material. The company produced 100 widgets in 35 days before they realised the widgets contained
radioactive material. Then it took 14 days before an investigation was started. The investigation took
90 days. During the investigation, a company worker would visit the location where a widget was
shipped to, and measured the radioactive material in the widget. Only one measurement of the
radioactive material was performed per widget.
In the accompanying decay.csv file you will find the time of measurement, and the amount of
radioactive material measured by the investigator, and some uncertainty in that measurement.
The company has asked you whether you can infer the rate of radioactive decay of the material in
the widgets. They know that the amount of radioactive material at any time t is
          N(t) = N_0 exp (−α [t − t0])
where N is measured in grams and t is measured in seconds. They also know that the detectors
they use to measure the amount of radioactive material reports uncertainties that are normally
distributed.
The company does not know how much radioactive material was initially put into each widget, but
they know that the amount is different for each widget. However, they do know that the initial
amount of radioactive material must be less than 20 grams in every widget.
Unfortunately, the company also does not know when each widget was made. All they can tell you is
that all 100 widgets were made in 35 days. You can see how this careless operation has led to such a
problematic situation!
Part 1
Infer the decay rate for the mysterious radioactive material.
Part 2
The company is delighted with your work. However, there is a problem. The investigation team used
three different detectors to measure the amount of radioactive material in each widget. Each
detector costs $1,000,000, and they are suspicious that one of those detectors is giving biased
measurements. They need you to infer which detector is giving biased measurements, and to infer
what the level of that bias is. After accounting for the biases, have your inferences on changed?
"""

########################################################################################################
#               Part 1
data = pd.read_csv('decay.csv')
t_measured = data['time_measured_in_seconds']
N_measured = data['grams_measured']
sigma_N_measured = data['uncertainty_in_grams_measured']
detector = data['detector_name']

def decay_model(alpha):
    return [20*np.exp(-alpha*(t*86400)) for t in range(140)]

d1 = decay_model(1e-5)
d2 = decay_model(1e-6)
d3 = decay_model(1e-7)
d4 = decay_model(1e-8)

# Plotting with error bars
plt.figure()
plt.errorbar(t_measured/86400, N_measured, yerr=sigma_N_measured, fmt='o')
plt.plot(np.linspace(50,140,90), d1[50:], label=r'$\alpha = 2e-5$')
plt.plot(np.linspace(50,140,90), d2[50:], label=r'$\alpha = 2e-6$')
plt.plot(np.linspace(50,140,90), d3[50:], label=r'$\alpha = 2e-7$')
plt.plot(np.linspace(50,140,90), d4[50:], label=r'$\alpha = 2e-8$')
plt.xlabel('Time (days)')
plt.ylabel('Mass (grams)')
plt.title('Mass of Radioactive Material')
plt.legend(loc=1, fontsize='x-small')
plt.show()
plt.savefig('Q1P1data.png')

######################  Define and read stan model  ####################### 

# Function to read the stan model from a file
def read_stan_model(file_path):
    with open(file_path, 'r') as file:
        stan_model = stan.CmdStanModel(stan_file=file_path)
    return stan_model


# Function to sample posteriors using the stan model
def sample_posteriors(stan_model_code, data, init):
    sm = stan.CmdStanModel(stan_file='decay2.stan')
    fit = sm.sample(data=data, iter_sampling=4000, chains=2, iter_warmup = 1000, inits = init)
    return fit

stan_model_path = 'decay.stan'
stan_model = read_stan_model(stan_model_path)

# define the required data
N_widgets = len(N_measured)
manufacturing_time = 35 * 86400
t_initial = np.random.uniform(0, manufacturing_time, size=(N_widgets, 1)).T[0]
N_initial_max = 20

data = dict(
    N_widgets=N_widgets,
    t_initial=t_initial,
    t_measured=t_measured,
    N_measured=N_measured,
    sigma_N_measured=sigma_N_measured,
    N_initial_max=N_initial_max,
)

inits = dict(alpha=2e-7)

optimised = stan_model.optimize(data=data, inits=inits)
params = optimised.stan_variables()
print(params)

fit = sample_posteriors(stan_model, data, params)

fig = fit.draws_pd(vars = 'alpha')
plt.figure()
plt.axhline(alpha, c='red', zorder=3)
plt.plot(fig)
plt.xlabel('draws')
plt.ylabel('alpha')
plt.title('Alpha Sampler Chain')
plt.show()

# Plot the kernel density estimate:
fig, ax = plt.subplots()
ax = sns.kdeplot(fit.draws_pd(vars="alpha"))
# Extract line:
kdeline = ax.lines[0]
xs = kdeline.get_xdata()
ys = kdeline.get_ydata()
# Mode of distribution:
mode_idx = np.argmax(ys)
mode_of_alpha = xs[mode_idx]
# Plot mode:
ax.vlines(mode_of_alpha, 0, ys[mode_idx], color='tomato', ls='--', lw=2, label='Mode of alpha')
# Plot optimized alpha:
ax.vlines(params['alpha'], 0, ys[mode_idx], color='purple', ls='--', lw=2, label='Optimized alpha')
ax.set_xlabel('Alpha')
ax.set_ylabel('Density')
plt.legend()
plt.title('Posterior Distribution of Alpha')
plt.show()
plt.savefig('q1p1alpha.png')
# Output the mode of alpha and the optimized value
print('Mode of alpha:', mode_of_alpha)
print('Optimized alpha:', params['alpha'])

init_mass = fit.draws_pd()

plt.figure()
for i in range(1,N_widgets+1):
    sns.kdeplot(init_mass[f'N_initial[{i}]'])
plt.xlabel('Initial mass')
plt.ylabel('frequency')
plt.title('N_initial')
plt.show()


########################################################################################################
#               Part 2

stan_model_path = 'decay2.stan'
stan_model = read_stan_model(stan_model_path)

# Mapping detectors to integers
mapping = {'A': 1, 'B': 2, 'C': 3}
detectors = detector.map(mapping).tolist()

# define the required data
N_widgets = len(N_measured)
manufacturing_time = 35 * 86400
t_initial = np.random.uniform(0, manufacturing_time, size=(N_widgets, 1)).T[0]
N_initial_max = 20
# print(detectors)

data = dict(
    N_widgets=N_widgets,
    t_initial=t_initial,
    t_measured=t_measured,
    N_measured=N_measured,
    sigma_N_measured=sigma_N_measured,
    N_initial_max=N_initial_max,
    detector = detectors
)

inits = dict(alpha=2e-7)

optimised = stan_model.optimize(data=data, inits=inits)
params = optimised.stan_variables()
print(params)

alpha = params['alpha']
print(alpha)

fit = sample_posteriors(stan_model, data, params)

fig = fit.draws_pd(vars = 'alpha')
plt.figure()
plt.axhline(alpha, c='red', zorder=3)
plt.plot(fig)
plt.xlabel('draws')
plt.ylabel('alpha')
plt.title('Alpha Sampler Chain')
plt.show()


# Plot the kernel density estimate:
fig, ax = plt.subplots()
ax = sns.kdeplot(fit.draws_pd(vars="alpha"))
# Extract line:
kdeline = ax.lines[0]
xs = kdeline.get_xdata()
ys = kdeline.get_ydata()
# Mode of distribution:
mode_idx = np.argmax(ys)
mode_of_alpha = xs[mode_idx]
# Plot mode:
ax.vlines(mode_of_alpha, 0, ys[mode_idx], color='tomato', ls='--', lw=2, label='Mode of alpha')
# Plot optimized alpha:
ax.vlines(params['alpha'], 0, ys[mode_idx], color='purple', ls='--', lw=2, label='Optimized alpha')
ax.set_xlabel('Alpha')
ax.set_ylabel('Density')
plt.legend()
plt.title('Posterior Distribution of Alpha')
plt.show()
# Output the mode of alpha and the optimized value
print('Mode of alpha:', mode_of_alpha)
print('Optimized alpha:', params['alpha'])

init_mass = fit.draws_pd()

plt.figure()
for i in range(1,N_widgets+1):
    sns.kdeplot(init_mass[f'N_initial[{i}]'])
plt.show()

# print(fit.detector_bias)

A_bias, B_bias, C_bias = fit.detector_bias.T
print(A_bias)

plt.figure()
plt.scatter(np.linspace(0,len(A_bias), len(A_bias)), A_bias, label='A Bias', s=4, alpha=0.8)
plt.scatter(np.linspace(0,len(B_bias), len(B_bias)), B_bias, label='B Bias', s=4, alpha=0.8)
plt.scatter(np.linspace(0,len(C_bias), len(C_bias)), C_bias, label='C Bias', s=4, alpha=0.8)
plt.xlabel('Widgets')
plt.ylabel('Bias')
plt.title('Biases of Detectors')
plt.legend()
plt.show()

