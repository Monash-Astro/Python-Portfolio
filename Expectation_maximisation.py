import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

############################################################################
#               Generate necessary data
def generate_data(data_seed, N=500, K=5):
    np.random.seed(data_seed)
    n_samples = np.random.multinomial(N, np.ones(K)/K)
    X, assignment = make_blobs(
        n_samples=n_samples,
        cluster_std=np.exp(np.random.uniform(-1, 0, size=K)),
        random_state=data_seed)
    return X

data_seed = 32481667
X = data = generate_data(data_seed)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=3, alpha=0.4)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()


def expectation(X, weights, pi, sigma, mu, dims, n_points, k):
    lh = np.zeros((n_points, k))
    for j in range(0,k,1):
        lh[:, j] = (pi[j] * (1 / np.sqrt((2 * np.pi) ** (dims) * np.linalg.det(sigma[j]))) * 
        np.exp( -0.5 * np.sum(np.dot((X - mu[j]), np.linalg.inv(sigma[j])) * (X - mu[j]), axis=1)))
    
    ll = np.sum(np.log(np.sum(lh, axis=1)))
    weights = lh / np.sum(lh, axis=1, keepdims=True)
    
    return ll, weights

def maximization(X, weights, pi, sigma, mu, n_points):
    num_k = np.sum( weights, axis=0 )
    for i, nk in enumerate(num_k):
        pi[i] = nk / n_points

        mu[i] = np.array(1/nk * np.sum( weights[:,i].reshape(-1,1) * X, axis=0))
        # print(mu)

        X_mid = X - mu[i]
        sigma[i] = 1/nk * (weights[:, i].reshape(-1, 1, 1) * np.einsum('ij,ik->ijk', X_mid, X_mid)).sum(axis=0)

    return pi, np.array(mu), sigma

########################################################################################################
#               Initialiser for data
def initialization():
    tol = 0.1   #set tolerance for execution
    k = 5
    pi = np.ones(k)/k
    sigma = np.zeros((k, 2, 2))
    for i in range(k):
        sigma[i] = np.eye(2)
    weights = np.zeros((len(X), k))
    mu = [[np.random.uniform(0, 2),np.random.uniform(0,2)] for _ in range(k)]
    colors = ['green', 'purple', 'red', 'blue', 'orange']

    return tol, k, pi, sigma, np.array(mu), weights, colors

########################################################################################################
#               Functions for plotting data with ellipses
def plot_g(mu, sigma, ax, n_std, color, facecolor='none'):
        a = sigma[0, 1]/np.sqrt(sigma[0, 0] * sigma[1, 1])
        r_x = np.sqrt(1 + a)
        r_y = np.sqrt(1 - a)
        e = Ellipse((0, 0), width=r_x * 2, height=r_y * 2,
                        facecolor=color, edgecolor=color, alpha=1/(n_std*4))
        scale_x = np.sqrt(sigma[0, 0]) * n_std
        mu_x = mu[0]
        scale_y = np.sqrt(sigma[1, 1]) * n_std
        mu_y = mu[1]
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mu_x, mu_y)
        e.set_transform(transf + ax.transData)
        return ax.add_patch(e)

def plot(title, mu, sigma, k, colors):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.4)
    ax.scatter(mu[:, 0], mu[:, 1], c=colors)
    for i in range(k):
        plot_g(mu[i], sigma[i], ax, 1, colors[i])
        plot_g(mu[i], sigma[i], ax, 2, colors[i])
        plot_g(mu[i], sigma[i], ax, 3, colors[i])
    
    plt.title(title)
    plt.savefig(title)
    plt.show()
    plt.clf()

########################################################################################################
#               EM Algorythim          (not very elegant but works)                          
def expectation_maximization(X):
    tol, k, pi, sigma, mu, weights, colors = initialization()

    step = 0
    plot('Step ' + str(step), mu, sigma, k, colors)

    lll,llsl = [], []
    ll, ll_last = 0,1
    while abs(ll_last - ll) > tol:
        step += 1
        ll_last = ll

        ll, weights = expectation(X, weights, pi, sigma, mu, 2, len(X), k)
        pi, mu, sigma = maximization(X, weights, pi, sigma, mu, len(X))

        lll.append(ll)
        llsl.append(abs(ll_last - ll))
        plot('Step ' + str(step), mu, sigma, k, colors)
    
    plt.figure()
    plt.plot(np.linspace(0, len(lll[1:]), len(lll[1:])), lll[1:])
    plt.xlabel('iterations')
    plt.ylabel('log-likelihood')
    plt.title('log-likelihood of data')
    plt.show()
    plt.savefig('ll.png')

    plt.figure()
    plt.plot(np.linspace(0, len(llsl[2:]), len(llsl[2:])),llsl[2:])
    plt.xlabel('iterations')
    plt.ylabel(r'$\Delta$ log-likelihood')
    plt.title(r'$\Delta$ log-likelihood of data')
    plt.show()
    plt.savefig('llsl.png')

expectation_maximization(X)
