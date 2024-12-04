import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 1.0  # Длина
T = 2.0  # Время
Nx = 100  # Количество пространственных делений
Nt = 500  # Количество временных шагов
c = 1.0  # Скорость распространения волны

dx = L / (Nx - 1)
dt = T / Nt

# Условие устойчивости
if c * dt / dx > 1:
    raise ValueError("Схема не устойчива: c * dt / dx должно быть <= 1.")

u = np.zeros((Nx, Nt))
x = np.linspace(0, L, Nx)

for i in range(Nx):
    u[i, 0] = 0.2 * (1 - x[i]) * np.sin(np.pi * x[i])


for i in range(1, Nx - 1):
    u[i, 1] = u[i, 0]


u[0, :] = 0
u[-1, :] = 0

# Явная схема
for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        u[i, n + 1] = (2 * u[i, n] - u[i, n - 1] +
                       (c * dt / dx) ** 2 * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n]))


fig, ax = plt.subplots()
line, = ax.plot(x, u[:, 0], color='blue')
ax.set_ylim(-0.2, 0.2)
ax.set_title('Волновое уравнение')
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')

def update(frame):
    line.set_ydata(u[:, frame])
    return line,

ani = FuncAnimation(fig, update, frames=Nt, blit=True)
plt.show()