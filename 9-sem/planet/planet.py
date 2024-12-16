import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Гравитационная постоянная
G = 6.67430e-11
# Масса Солнца
M = 1.989e30

# Параметры планет
planet = [
    [3.3e23, 4.87e24, 5.97e24, 6.42e23], # масса 
    [0.39 * 1.496e11, 0.72 * 1.496e11, 1.0 * 1.496e11, 1.52 * 1.496e11], # расстояние
    [47.87e3, 35.02e3, 29.78e3, 24.07e3], # начальная скорость
]

# Условия начального положения и скорости
positions = np.array([[d, 0] for d in planet[1]])
velocities = np.array([[0, v] for v in planet[2]])

# Время симуляции
dt = 3600  # шаг времени в секундах (1 час)
steps = 365 * 24  # количество шагов (1 год)

# Для хранения координат для анимации
trajectory = [[] for _ in range(len(planet[0]))]

# Основной цикл по времени
for step in range(steps):
    for i in range(len(planet[0])):
        r = np.linalg.norm(positions[i])
        # Ускорение
        a = -G * M / r**2
        # Обновление скорости
        velocities[i] += a * (positions[i] / r) * dt
        # Обновление позиции
        positions[i] += velocities[i] * dt
        # Сохранение пути
        trajectory[i].append(positions[i].copy())

# Настройка графики
fig, ax = plt.subplots()
ax.set_xlim(-2 * 1.496e11, 2 * 1.496e11)
ax.set_ylim(-2 * 1.496e11, 2 * 1.496e11)
ax.set_aspect('equal')
ax.set_title('Движение планет вокруг Солнца')

sun = plt.Circle((0, 0), 1.5e10, color='yellow')
ax.add_artist(sun)

lines = [ax.plot([], [], marker='o')[0] for _ in planet[0]]

def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(frame):
    for i, line in enumerate(lines):
        line.set_data([pos[0] for pos in trajectory[i][:frame+1]], 
                        [pos[1] for pos in trajectory[i][:frame+1]])
    return lines

ani = FuncAnimation(fig, update, frames=len(trajectory[0]), init_func=init, blit=True, interval=30)
plt.show()