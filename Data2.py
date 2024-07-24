
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as op
from scipy.stats import norm
from chainconsumer import Chain, ChainConsumer
import pandas as pd
from corner import corner
import stan_utils as Stan



# ########################################################################################################################
# #               Question 1
# ########################################################################################################################

# # Function to calculate Rosenbrock values
# def rosenbrock(theta, a=1, b=100):
#     x, y = theta
#     return (a - x)**2 + b * (y - x**2)**2

# # 250-250 grid
# x = np.linspace(-2, 2, 250)
# y = np.linspace(-2, 2, 250)
# theta = np.meshgrid(x, y)

# # Calculate the Rosenbrock function values for each point in the grid
# cont = np.log(rosenbrock(theta))

# # Plot the contours
# plt.figure()
# plt.contour(theta[0], theta[1], cont, levels=50, cmap='RdPu')
# plt.colorbar(label='Rosenbrock Value')
# plt.title('Rosenbrock Contour Plot')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
# plt.savefig('q1_figure.png')



# ########################################################################################################################
# #               Question 2
# ########################################################################################################################

# # Initial position
# x0 = np.array([-1.0, -1.0])

# # Function to record the positions tried by the optimizer
# def objective_function(theta):
#     temp_pos.append(theta)

# # BFGS (or L-BFGS-B) algorithm
# temp_pos=[np.array([-1.0,-1.0])]
# op_result_bfgs = op.minimize(rosenbrock, x0, method='L-BFGS-B', callback=objective_function)
# BFGS_pos = temp_pos

# # Nelder-Mead algorithm
# temp_pos=[np.array([-1.0,-1.0])]
# op_result_nelder_mead = op.minimize(rosenbrock, x0, method='Nelder-Mead', \
#                                     callback=objective_function)
# NM_pos = temp_pos

# # TNC's algorithm
# temp_pos=[np.array([-1.0,-1.0])]
# op_result_TNC = op.minimize(rosenbrock, x0, method='TNC', callback=objective_function)
# TNC_pos = temp_pos

# # Plot the Rosenbrock function contours
# x = np.linspace(-2, 2, 250)
# y = np.linspace(-2, 2, 250)
# X, Y = np.meshgrid(x, y)
# Z = rosenbrock([X, Y])
# plt.contour(X, Y, Z, levels=50, cmap='RdPu')

# # Plot the optimization paths
# plt.plot(*zip(*BFGS_pos), marker='o', label='BFGS')
# plt.plot(*zip(*NM_pos  ), marker='o', label='Nelder-Mead')
# plt.plot(*zip(*TNC_pos ), marker='o', label='dogleg')

# # Mark the final solution
# plt.scatter(op_result_bfgs.x[0], op_result_bfgs.x[1], color='red', \
#             marker='x', label='BFGS Solution')
# plt.scatter(op_result_nelder_mead.x[0], op_result_nelder_mead.x[1], color='blue', \
#             marker='x', label='Nelder-Mead Solution')
# plt.scatter(op_result_TNC.x[0], op_result_TNC.x[1], color='green', marker='x', \
#             label='TNC Solution')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Optimization Paths for Rosenbrock Function')
# plt.legend()
# plt.show()
# plt.savefig('q2_figure.png')



# ########################################################################################################################
# #               Question 3
# ########################################################################################################################

# # Create fake data
# np.random.seed(0)
# N = 30
# x = np.random.uniform(0, 100, N)
# theta = np.random.uniform(-1e-3, 1e-3, size=(4, 1))
# # Define the design matrix.
# A = np.vstack([
#     np.ones(N),
#     x,
#     x**2,
#     x**3
# ]).T

# y_true = (A @ theta).flatten()
# y_err_intrinsic = 10 # MAGIC number!
# y_err = y_err_intrinsic * np.random.randn(N)

# y = y_true + np.random.randn(N) * y_err
# y_err = np.abs(y_err)

# ################################################################################################

# # Models with different numbers of parameters
# Y = np.atleast_2d(y).T
# model_orders = [1, 2, 3, 4, 5, 6]
# bics = []
# bic_values = []
# C = np.diag(y_err * y_err)
# C_inv = np.linalg.inv(C)


# for n in model_orders:
#     # Modify design matrix A to include polynomial terms up to the given degree
#     A = np.vstack([x**d for d in range(n)]).T
    
#     # Calculate parameters
#     G = np.linalg.inv(A.T @ C_inv @ A)
#     X = G @ (A.T @ C_inv @ Y)
    
#     # Compute log likelihood
#     l = len(Y)
#     U = Y - A @ X
#     log_likelihood = -0.5 * U.T @ np.linalg.inv(C) @ U

#     # Calculate BIC
#     bic = -2 * log_likelihood + n * np.log(l)
#     bic_values.append(bic.flatten().flatten())



# print(bic_values)
# print(np.argmin(bic_values))

# # Plot BIC as a function of number of parameters
# plt.figure(figsize=(10, 6))
# plt.plot(model_orders, bic_values, marker='o', linestyle='-', color = 'r')
# plt.xlabel('Number of Parameters (theta)')
# plt.ylabel('Bayesian Information Criterion')
# #plt.title('Bayesian Information Criteria (BIC)')
# plt.grid(True)
# plt.show()
# plt.savefig('q3figure.png')


########################################################################################################################
#               Question 4
########################################################################################################################


def U(x_mean, y_mean):
    z_x = (x - x_mean) / sig_x
    z_y = (y - y_mean) / sig_y
    U = (z_x**2 + z_y**2) + np.log(2 * np.pi * sig_x * sig_y)
    return np.sum(U)


def dU_dx(x_mean, y_mean):
    dU_dxm = 0
    dU_dym = 0
    for i in range(len(x)):
        dU_dxm += -2 * (x[i] - x_mean)/(sig_x[i]**2)
        dU_dym += -2 * (y[i] - y_mean)/(sig_y[i]**2)
    return np.array([dU_dxm, dU_dym])


########################################################################################################################
#               Question 5
########################################################################################################################

                   #  x     y    sig_x sig_y
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


def leapfrog_integration(x, p, dU_dx, n_steps, step_size):

    xp = [] 
    yp = []
    E  = []
    pp = []

    x = np.copy(x)
    p = np.copy(p)

    xp.append(x[0])
    yp.append(x[1])
    E.append(U(*x))
    pp.append(0.5*p.T@p)

    # Take a half-step first.
    p -= 0.5 * step_size * dU_dx(*x)
    for step in range(n_steps):
        x += step_size * p
        p -= step_size * dU_dx(*x)

        xp.append(x[0])
        yp.append(x[1])
        E.append(U(*x))
        pp.append(0.5*p.T@p)

    x += step_size * p
    p -= 0.5 * step_size * dU_dx(*x)

    xp.append(x[0])
    yp.append(x[1])
    E.append(U(*x))
    pp.append(0.5*p.T@p)


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
plt.savefig('q_5figure.png')

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
plt.savefig('q_5figure2.png')



########################################################################################################################
#               Question 6
########################################################################################################################

from tqdm import tqdm

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
plt.savefig('q_6figure.png')


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
plt.savefig('q_6figure2.png')


df1 = pd.DataFrame(thetas1, columns=['x', 'y'])
df2 = pd.DataFrame(thetas2, columns=['x', 'y'])

c = ChainConsumer()
c.add_chain(Chain(samples=df1, name="Chain 1"))
c.add_chain(Chain(samples=df2, name="Chain 2"))
fig = c.plotter.plot()
fig.savefig('q_6figure3.png')







########################################################################################################################
#               Question 7
########################################################################################################################

# model = Stan.load_stan_model("HMC.stan")


# data_dict = dict(
#     x = x,
#     y = y,
#     sig_x = sig_x,
#     sig_y = sig_y    
# )

# # Run optimisation.
# opt_stan = model.optimizing(
#     data=data_dict
# )

# # Run sampling.
# samples = model.sampling(**stan.sampling_kwds(
#     chains=2,
#     iter=1000,
#     data=data_dict,
#     init=opt_stan
# ))


# print(samples)











