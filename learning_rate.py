# didactic comparison between learning rates
import numpy as np
import matplotlib.pyplot as plt


def f_to_min(x):
    return (x - 10.)**2 + 5.


def df_to_min(x):
    return 2. * (x - 10.)


lr = [0.1, 0.9, 1.1]
pos_init = [0., 0., 7.]
chemin = np.zeros((3, 20))
for k in range(3):
    chemin[k, 0] = pos_init[k]
    for i in range(1, 20):
        chemin[k, i] = chemin[k, i-1] - lr[k] * df_to_min(chemin[k, i-1])


lx = np.linspace(0., 20., 100, endpoint=True)
ly = f_to_min(lx)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.plot(lx, ly, 'b', linewidth=2)
ax1.plot(chemin[0], f_to_min(chemin[0]), 'ro--', linewidth=2, markersize=8)
ax1.text(5., 100., 'learning rate = '+str(lr[0]))
# ax1.set_title("2D linear regression", fontsize=20)
# ax1.set_xlabel("X", fontsize=12)
# ax1.set_ylabel("Y", fontsize=12)
# ax1.tick_params(axis='both', which='major', labelsize=10)
# ax1.grid(color='gray', linestyle='--', linewidth=0.5)
ax2.plot(lx, ly, 'b', linewidth=2)
ax2.plot(chemin[1], f_to_min(chemin[1]), 'ro--', linewidth=2, markersize=8)
ax2.text(5., 100., 'learning rate = '+str(lr[1]))
ax3.plot(lx, ly, 'b', linewidth=2)
ax3.plot(chemin[2,:7], f_to_min(chemin[2,:7]), 'ro--', linewidth=2, markersize=8)
ax3.text(5., 100., 'learning rate = '+str(lr[2]))
plt.show()

