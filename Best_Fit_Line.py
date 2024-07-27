import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
import scipy.optimize as op
import emcee
from corner import corner
from matplotlib.ticker import MaxNLocator

"""
Project decription
---------------------------------------------------------------------------------------------
This file generates a set of data points from a straight line, with uncertainties in the x 
and y directions, as well as covariances between the uncertainties.
I then determine an estimate best fit using linear algebra methods, and use this
estimate to determine the log prior probability distribution, log likelihood, and log
posterior probability distribution.
I am then able to sample model posteriors using emcee sampling. These samples are displayed
in a corner plot showing the marginalized posterior distributions.
Finally I plot the model posterior predictions against the data and the initial guess, with 
each data point coloured by the posterior probability that it is an outlier.
---------------------------------------------------------------------------------------------
"""

############################################################################
#       Generate data points and random 2D Covariances

name = 'Ben Charles'
data_code = 10001001

def generate_data(code):
    
    N = 20
    x_low, x_high = (0, 500)
    y_low, y_high = (0, 1000)
    
    np.random.seed(code)
    
    m = np.tan(np.arcsin(np.random.uniform(-1, 1)))
    
    x_transform = lambda x: x * (x_high - x_low) + x_low
    y_transform = lambda y: y * (y_high - y_low) + y_low
    x_true = x_transform(np.random.uniform(size=N))
    b = -1.1 * np.min(m * x_true) + np.random.normal(0, 1)
    y_true = m * x_true + b
    
    y_err_scale = 5 * (y_high - y_low) / 100
    x_err_scale = 5 * (x_high - x_low) / 50
    x_err = x_err_scale * np.random.randn(N)
    y_err = y_err_scale * np.random.randn(N)
    
    x = x_true + x_err * np.random.randn(N)
    y = y_true + y_err * np.random.randn(N)
    x_err, y_err = (np.abs(x_err), np.abs(y_err))
    rho_xy = np.random.uniform(low=-1, high=+1, size=N)
    
    N_outliers = np.random.randint(1, 5)
    outlier_indices = np.random.choice(N, N_outliers, replace=False)
    x[outlier_indices] = x_transform(np.random.uniform(size=N_outliers))
    y[outlier_indices] = y_transform(np.random.uniform(size=N_outliers))
    
    return np.vstack([x, y, x_err, y_err, rho_xy]).T


data = generate_data(data_code)
np.savetxt("data.csv", data)

# Load required data
x, y, x_err, y_err, rho_xy = np.loadtxt("data.csv").T

############################################################################
#       Plot the data ; with x-y-errors/covariances

def _ellipse(x, y, cov, scale=2, **kwargs):
    vals, vecs = np.linalg.eig(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
    w, h = scale * np.sqrt(vals)

    kwds = dict(lw=0.5, color='k')
    kwds.update(**kwargs)

    ellipse = Ellipse(xy=[x, y], 
                      width=w, height=h, angle=theta,
                      **kwds)
    ellipse.set_facecolor("none")
    return ellipse

covs = np.array([[[x_e**2, x_e*y_e*rho],
                  [x_e*y_e*rho, y_e**2]] \
                  for y_e, x_e, rho in zip(*data.T[2:])])

fig, ax = plt.subplots(figsize=(6, 6))
    
ax.scatter(x, y, c="k", s=10)

for xi, yi, cov in zip(x, y, covs):
    ax.add_artist(_ellipse(xi, yi, cov))
ax.plot(0,0,c='blue', label='est. BFL (lin-alg)')
    
# ax.set_xlim(-50, 650)
# ax.set_ylim(-100, 350)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title('Generated data - w. error ellipse')
    
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(7))
    
fig.tight_layout()
fig.legend()
fig.savefig('Generated data.png')
plt.show()

############################################################################
#       initialize model parameters

Y = np.atleast_2d(y).T

A = np.vstack([np.ones_like(x), x]).T

#create values for a rough linear alg solution
C = np.diag(y_err * y_err)

C_inv = np.linalg.inv(C)
G = np.linalg.inv(A.T @ C_inv @ A)
X = G @ (A.T @ C_inv @ Y)

#find rough solution for theta using linear alg
b_lin, m_lin = X.T[0]
y_med = np.median(y)
print('The median of y is:', y_med, '\n')
print('The results for b and m using linear algebra are:', '\n', 'm: ', m_lin, '\n', 'b: ', b_lin)
#plot rough solution
xl = np.array([-100, 700])
ax.plot(xl, xl * m_lin + b_lin, 
        "-", c="tab:blue",
        lw=2, zorder=-1)

# Construct the covariance matrices that we'll actually need.
C = np.array([[[x_e**2, x_e*y_e*rho],[x_e*y_e*rho, y_e**2]] \
              for y_e, x_e, rho in zip(y_err, x_err, rho_xy)])

C_inv = np.linalg.inv(C)


############################################################################
#       Define functions for prior, likelihood, and probability

def log_prior(theta):
    b, m, Q, mu_o, ln_sigma_v = theta
    if not (1 > Q > 0) \
    or not (700 > mu_o > -200) \
    or not (5 > ln_sigma_v > -5):
        return -np.inf
    return -3/2 * np.log(1 + m**2)


# We will describe the straight line model as 'foreground'.
def log_likelihood_fg(theta, x, y, C):
    b, m, Q, mu_o, ln_sigma_v = theta
    
    # Define projection vector.
    V = np.array([[-m, 1]]).T

    Delta = (y - m * x - b)
    Sigma = (V.T @ C @ V).flatten()
    
   # return -0.5 * (Delta**2 / Sigma**2) - 0.5 * Sigma**2 \
   #        -0.5*np.log(2*np.pi)
    return -0.5*np.log(Sigma) - 0.5*Delta**2/Sigma


def log_likelihood_bg(theta, x, y, C):
    b, m, Q, mu_o, ln_sigma_v = theta
    x_err = [c[0,0] for c in C]
    y_err = [c[1,1] for c in C]

    total_variance = y_err + np.exp(ln_sigma_v)**2
    return -0.5 * ((y - mu_o)**2 / total_variance) - 0.5 * np.log(total_variance)

def log_likelihood(theta, x, y, C):    
    b, m, Q, mu_o, ln_sigma_v = theta
    # Compute weighted foreground likelihoods for each data point.
    # (They are weighted by the log(Q) term)
    ll_fg = np.log(Q) + log_likelihood_fg(theta, x, y, C)
    
    # Compute weighted background likelihoods for each data point.
    ll_bg = np.log(1 - Q) + log_likelihood_bg(theta, x, y, C)

    # Sum the log of the sum of the exponents of the log likelihoods.
    ll = np.sum(np.logaddexp(ll_fg, ll_bg))
    v_stack = np.vstack([ll_fg, ll_bg])

    return (ll, v_stack)


def log_posterior(theta, x, y, C):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return (-np.inf, np.nan * np.ones((2, x.size)))

    ll, v_stack = log_likelihood(theta, x, y, C)
    return (lp + ll, v_stack)


############################################################################
#       Optimize the linear algebra solution
args = (x, y, C)
initial_theta = np.array([b_lin, m_lin, 0.8, y_med, -3])

result = op.minimize(
    lambda *args: -log_posterior(*args)[0],
    initial_theta,
    args=args,
    method="L-BFGS-B",
    bounds=[
        (None, None),
        (None, None),
        (0, 1),
        (-200, 700),
        (-5, 5)
    ]
)

print('The results for b and m using L-BFGS-B optimization are:', \
 '\n', 'm: ', result.x[1], '\n', 'b: ', result.x[0])

# Sample!
ndim, nwalkers = (result.x.size, 64)
p0 = [result.x + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]

sampler = emcee.EnsembleSampler(
    nwalkers, 
    ndim,
    log_posterior,
    args=args
)

# Run the burn-in.
pos, *_ = sampler.run_mcmc(p0, 1000)
sampler.reset()

# Run production.
sampler.run_mcmc(pos, 2000)

# Make a corner plot just showing m, b, Q
# (the others are nuisance parameters to us)
chain = sampler.chain.reshape((-1, ndim))

fig = corner(
    chain[:, :3],
    labels=(r"$b$", r"$m$", r"$Q$"),
)

############################################################################
#       Calculate posterior probability estimates for each data point.
q = np.zeros_like(x)
for i in range(sampler.chain.shape[1]):
    for j in range(sampler.chain.shape[0]):
        ll_fg, ll_bg = sampler.blobs[i][j]
        q += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
  
q /= np.prod(sampler.chain.shape[:2])


############################################################################
#       Plot data with posterior probability estimates

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x, y, c=q, edgecolor="k", lw=1, cmap="Greys", s=20)
# We are using arbitrary covariance matrices, so we should show 
# error ellipses instead of error bars.
for xi, yi, cov in zip(x, y, covs):
    ax.add_artist(_ellipse(xi, yi, cov, scale=2, color="k"))
ax.plot(0,0,c='blue', label='est. BFL (lin-alg)')


xlim = np.array([-100, 650])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
    
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(7))
    
# Plot draws of the posterior.
for index in np.random.choice(chain.shape[0], size=100):
    b, m = chain[index, :2]
    ax.plot(
        xlim,
        m * xlim + b,
        "-",
        c="tab:orange",
        alpha=0.2,
        lw=0.5,
        zorder=-100
    )

ax.plot(xl, xl * m_lin + b_lin, 
    "-", c="tab:blue",
    lw=2, zorder=-1)
ax.plot(0,0,c='orange', label='Posterior Draws')

ax.set_xlim(*xlim)
ax.set_ylim(-100, 400)
ax.set_title('Data w. identified outliers')
fig.legend()
fig.tight_layout()

