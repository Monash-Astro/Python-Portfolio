
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as op
from scipy.stats import norm
from chainconsumer import Chain, ChainConsumer
import pandas as pd
from corner import corner

"""
This program assumes a 2D model with uncertainties in both dimensions.
Within I have constructed a Hamiltonian Monte Carlo Sampler in order to sample the phase
space of this 2D model. By assuming the model depicts a physical gravity well, the phase
space can be sampled by 'shooting' a test particle through the model.
At time a particle in the system at location x(t) with momentum p(t) can be fully 
described by the Hamiltonian H(x, p) = U(x) + K(p).
The time evolution of the particle is then described by dx/dt = δH/δp = p and 
dp/dt = −δH/δx = −δlogp(x)/δx where logp(x) is the log posterior probability of the
model pararameters x given y. The Hamiltonian monte-carlo sampler makes use of this 
potential and momentum distribution by sampling the model space as if it were a particle 
caught in a gravity well.
"""

###########################################################################################
#                    Define required data

x, y, sig_x, sig_y = data = \
          np.array([[0.38, 0.32, 0.26, 0.01],
                    [0.30, 0.41, 0.07, 0.02],
                    [0.39, 0.25, 0.09, 0.04],
                    [0.30, 0.39, 0.07, 0.10],
                    [0.19, 0.32, 0.23, 0.02],
                    [0.21, 0.37, 0.15, 0.01],
                    [0.28, 0.31, 0.01, 0.06],
                    [0.24, 0.32, 0.02, 0.06],
                    [0.35, 0.29, 0.15, 0.02],
                    [0.23, 0.26, 0.15, 0.02]]).T

# energy hamiltonian function
def U(x_mean, y_mean):
    z_x = (x - x_mean) / sig_x
    z_y = (y - y_mean) / sig_y
    U = (z_x**2 + z_y**2) + np.log(2 * np.pi * sig_x * sig_y)
    return np.sum(U)

# energy hamiltonian derivative function
def dU_dx(x_mean, y_mean):
    dU_dxm = 0
    dU_dym = 0
    for i in range(len(x)):
        dU_dxm += -2 * (x[i] - x_mean)/(sig_x[i]**2)
        dU_dym += -2 * (y[i] - y_mean)/(sig_y[i]**2)
    return np.array([dU_dxm, dU_dym])

# terrible leapfrog integrator
def leapfrog_integration(x, p, dU_dx, n_steps, step_size):
    xp, yp, E, pp = [x[0]], [x[1]], [U(*x)], [0.5 * p.T @ p]
    x, p = np.copy(x), np.copy(p)
    p -= 0.5 * step_size * dU_dx(*x)
    
    for step in range(n_steps):
        x += step_size * p
        p -= step_size * dU_dx(*x)
        xp.append(x[0])
        yp.append(x[1])
        E.append(U(*x))
        pp.append(0.5 * p.T @ p)
    
    x += step_size * p
    p -= 0.5 * step_size * dU_dx(*x)
    xp.append(x[0])
    yp.append(x[1])
    E.append(U(*x))
    pp.append(0.5 * p.T @ p)
    
    return x, -p, xp, yp, E, pp

x_m = np.mean(x)
y_m = np.mean(y)
theta = np.array([x_m,y_m])

p1 = np.random.normal(0,1,2)
p2 = np.random.normal(0,1,2)
p3 = np.random.normal(0,1,2)

x1_prime, p1_prime, x1, y1, E1, pp1 = leapfrog_integration(theta, p1, dU_dx, 1000, 5e-5)
x2_prime, p1_prime, x2, y2, E2, pp2 = leapfrog_integration(theta, p2, dU_dx, 1000, 5e-5)
x3_prime, p3_prime, x3, y3, E3, pp3 = leapfrog_integration(theta, p3, dU_dx, 1000, 5e-5)


plt.figure()
plt.scatter(x1, y1, s=0.5, c='b')
plt.scatter(x2, y2, s=0.5, c='r')
plt.scatter(x3, y3, s=0.5, c='g')
plt.title('leapfrog')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.figure()
plt.plot([i for i in range(len(E1))],np.array(E1), c='blue', label='Potential')
plt.plot([i for i in range(len(E1))],np.array(pp1), c='red', label ='Kinetic')
plt.plot([i for i in range(len(E1))],np.array(pp1) +np.array(E1), c='orange', label='Total')
# plt.plot([i for i in range(len(E2))],np.array(pp2) +np.array(E2), c='blue', label='Total')
# plt.plot([i for i in range(len(E3))],np.array(pp3) +np.array(E3), c='green', label='Total')
plt.title('Energy over integration')
plt.xlabel('time')
plt.ylabel('Energy')
plt.legend()
plt.show()


from tqdm import tqdm

# Create the hamiltonian integrator
def ham_int(x_init, y_init, sig_x, sig_y, integrator, U, dU_dx):

    # theta = np.array([np.mean(x),np.mean(y)])
    theta = np.array([x_init,y_init])
    thetas = [[x_init,y_init]]
    lines = []

    for i in tqdm(range(1000)):
        p = np.random.normal(0,1,2)
        x_prime, p_prime, x, y, E, pp = leapfrog_integration(theta, p, dU_dx, 500, 5e-5)

        alpha = np.exp(-U(*x1_prime) + U(*theta) -pp[-1] + pp[0])
        u = np.random.uniform(0,1)
        if alpha >= u:
            theta = x1_prime
            thetas.append(x_prime)
            lines.append([x,y])
        else:
            thetas.append(x_prime)
            lines.append([x,y])
        # print(lines)
    return (np.array(thetas), np.array(lines))


for i in range(2):
    x_init, y_init = np.random.uniform(0.3,0.5,2)
    thetas1, lines1 = ham_int(x_init, y_init, sig_x, sig_y, leapfrog_integration, U, dU_dx)

    x_init, y_init = np.random.uniform(0.3,0.5,2)
    thetas2, lines2 = ham_int(x_init, y_init, sig_x, sig_y, leapfrog_integration, U, dU_dx)

thetas1 = thetas1[50:1000]
thetas2 = thetas2[50:1000]

plt.figure()
for i in range(len(lines1)):
    plt.scatter(lines1[i][0], lines1[i][1], s=0.5)
    plt.scatter(lines2[i][0], lines2[i][1], s=0.5)
plt.title('leapfrog')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


fig, axes = plt.subplots(2, 1, figsize=(8, 4))
labels = ('x', 'y')
for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(thetas1[:, i], lw=1, c='red')
    ax.plot(thetas2[:, i], lw=1, c='orange')
    # ax.axhline(theta[i], c="tab:blue", lw=2)
    ax.set_xlabel(r"Step")
    ax.set_ylabel(label)
    # So we can see the initial behaviour:
    ax.set_xlim(-10, 1000)
    fig.tight_layout()


df1 = pd.DataFrame(thetas1, columns=['x', 'y'])
df2 = pd.DataFrame(thetas2, columns=['x', 'y'])

c = ChainConsumer()
c.add_chain(Chain(samples=df1, name="Chain 1"))
c.add_chain(Chain(samples=df2, name="Chain 2"))
fig = c.plotter.plot()


