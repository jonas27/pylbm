import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation

x_dim = 300
y_dim = 300
epochs = 100000
omega = 1.7
top_vel = 0.1
path = "./vels.npy"
velocities = np.load(path)

rc("animation", html="html5")


def animate(t):
    v = velocities[t]
    ax.clear()
    # ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 0.5, color="black", linewidth=3.0)
    ax.plot(np.arange(x_dim), np.zeros((x_dim)) + y_dim - 1.5, color="orange", linewidth=5.0)
    ax.plot(np.arange(x_dim), np.zeros((x_dim)) + 0.5, color="black", linewidth=3.0)
    ax.plot(np.zeros(y_dim) + 0.5, np.arange((y_dim)), color="black", linewidth=3.0)
    ax.plot(np.zeros(y_dim) + x_dim - 1.5, np.arange((y_dim)), color="black", linewidth=3.0)
    ax.axis("equal")
    strm = ax.streamplot(np.arange(x_dim), np.arange(y_dim), v[0, :, :].T, v[1, :, :].T, cmap="autumn")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


fig, ax = plt.subplots()
# framesiterable, int, generator function, or None, optional -- Source of data to pass func and each frame of the animation
# intervalint, default: 200 -- Delay between frames in milliseconds.
ani = FuncAnimation(fig, animate, frames=100, interval=300, repeat=False)
# writervideo = animation.FFMpegWriter(fps=1)
ani.save("./m6.gif", writer="imagemagick", fps=30)
plt.close()