# Compute the linear regression on a cluster of 2D points
import math
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


class Reglin:
    """ Encapsulates data and methods for the linear regression"""
    def __init__(self, xm, ym, thv, alpha, nbit):
        """
        :param xm: features matrix (nb_items X nb_features+1)
        :param ym: target values (nb_items X 1)
        :param thv: model's parameters (nb_features + 1)
        :param alpha: learning rate
        :param nbit: number of iterations of the gradient descent
        """
        self.caract = xm
        self.cible = ym
        self.param = thv
        self.step = alpha
        self.nb_iter = nbit
        self.cost_hist = []

    def cost(self, p=[]):
        """
        Compute de cost function. If 'p' is not empty, uses 'p' as the parameters instead of 'self.params'.
        """
        if len(p) > 0:
            cost = np.sum(np.square(self.caract @ p - self.cible)) / (2 * self.cible.size)
        else:
            cost = np.sum(np.square(self.caract @ self.param - self.cible)) / (2 * self.cible.size)
        return cost

    def grad_cost(self):
        """
        Compute the derivatives od the cost function with respect to each parameter
        """
        return (self.caract.transpose() @ (self.caract @ self.param - self.cible)) / self.cible.size

    def grad_descent(self):
        """ Performs the parameters optimization by gradient descent.
        Check the cost value and adapt the learning rate if the cost doesn't decrease.
        Save the evolution of the cost function during the optimization in 'cost_hist'."""
        i = 0
        while i < self.nb_iter:
            cost = self.cost()
            if i == 0:
                self.cost_hist.append(cost)
            grad = self.grad_cost()
            newparam = self.param - self.step * grad
            newcost = self.cost(newparam)
            if newcost <= cost:
                self.cost_hist.append(newcost)
                self.param = newparam
                i += 1
            else:
                self.step /= 2.
        self.cost_hist.append(self.cost())


# data creation
# -------------

def linear_points_generation(nb, x_min, x_max, a, b, sigma):
    """ Create a cloud of 'nb' points in 2D with abscissa uniformly distributed
    between 'x_min' and 'x_max. The y coordinates are randomly distributed
    according a normal distribution with a standard deviation 'sigma'
    around a linear function.
    :return: two vectors, one with the x values and one with the y values"""
    vx = (x_max - x_min) * np.random.random_sample(nb) + x_min
    vy = a * vx + b + np.random.randn(nb) * sigma
    return vx, vy


# data visualization
# ------------------

nb_points = 100
beta_0 = 5.
beta_1 = 0.8
x_values, y_values = linear_points_generation(nb_points, 0., 10., beta_1, beta_0, 1.)
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.scatter(x_values, y_values, c='red', marker='+', s=50, linewidths=1.)
ax1.set_title("2D cluster", fontsize=20)
ax1.set_xlabel("X", fontsize=12)
ax1.set_ylabel("Y", fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.grid(color='gray', linestyle='--', linewidth=0.5)


# Creation of the features matrix and targets matrix
# --------------------------------------------------

xmat = np.array([np.ones(nb_points), x_values]).transpose()
ymat = np.reshape(y_values, (nb_points, -1))
theta = np.zeros((2, 1))


# linear regression
# -----------------

nb_iter = 5000
learning_rate = 0.01
one_reglin = Reglin(xmat, ymat, theta, learning_rate, nb_iter)
one_reglin.grad_descent()
print("final parameters: ", one_reglin.param)
print("final cost: ", one_reglin.cost_hist[nb_iter])


# Computation by normal equations
#--------------------------------

beta_norm = linalg.pinvh(xmat.transpose() @ xmat) @ xmat.transpose() @ ymat
print("normal parameters: ", beta_norm)

# results visualization
# ---------------------

# lidx = list(range(20))
# ax2.plot(lidx, one_reglin.cost_hist[0:20], c='green', linewidth=2)
# ax2.set_ylim((0., one_reglin.cost_hist[0]*1.1))
# ax2.set_title("Cost function history", fontsize=20)
# ax2.set_xlabel("number of iterations", fontsize=12)
# ax2.set_ylabel("Cost", fontsize=12)
# ax2.tick_params(axis='both', which='major', labelsize=10)
# ax2.grid(color='gray', linestyle='--', linewidth=0.5)

ax1.plot(x_values, xmat @ one_reglin.param, c='blue', linewidth=1)
plt.show()
