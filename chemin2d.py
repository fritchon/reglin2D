# illustration du cheminement avec gradient descent sur une fonction z = f(x,y)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def f_to_min(x, y):
    return np.cos(x) + np.cos(2 * y)


def df_to_min(x, y):
    return np.array([- np.sin(x), - 2. * np.sin(2. * y)])


lr = 0.2
nb_iter = 12
pos_init = np.array([1., 0.5])
chemin = np.zeros((nb_iter, 2))
grads = np.zeros((nb_iter, 2))
chemin[0] = pos_init
for i in range(1, nb_iter):
    grads[i-1] = df_to_min(chemin[i-1, 0], chemin[i-1, 1])
    chemin[i] = chemin[i-1] - lr * grads[i-1]

print(chemin[:, 0], chemin[:, 1])
print(f_to_min(chemin[:, 0], chemin[:, 1]))
nx, ny = (100, 100)
lx = np.linspace(0., 5., nx)
ly = np.linspace(0., 3., ny)
xm, ym = np.meshgrid(lx, ly)
zm = f_to_min(xm, ym)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(xm, ym, zm, cmap=cm.coolwarm, alpha=0.5, linewidth=0, antialiased=False)
surf = ax.plot_wireframe(xm, ym, zm, linewidth=0.4)
ax.plot(chemin[:, 0], chemin[:, 1], f_to_min(chemin[:, 0], chemin[:, 1]), 'go--', linewidth=2, markersize=6)
ax.quiver(chemin[:, 0], chemin[:, 1], f_to_min(chemin[:, 0], chemin[:, 1]), 0.5 * grads[:, 0], 0.5 * grads[:, 1], 0., color='red')
plt.show()

plt.contourf(xm, ym, zm, 50, cmap=cm.coolwarm)
plt.axis('scaled')
plt.plot(chemin[:, 0], chemin[:, 1], 'go--', linewidth=2, markersize=8)
plt.quiver(chemin[:, 0], chemin[:, 1], grads[:, 0], grads[:, 1], color='red', width=0.004)
plt.show()