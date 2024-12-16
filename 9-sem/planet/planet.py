import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры
G = 6.67430e-11  # универсальная гравитационная постоянная
m_sun = 1.989e30  # масса Солнца
time_duration = 365 * 24 * 3600  # Сколько секунд моделируем
dt = 60 * 60  # Шаг по времени (1 час)

# Начальные условия (планеты: масса, x, y, vx, vy)
planets = [
    {'name': 'Меркурий', 'mass': 3.30e23, 'x': 0.39 * 1.496e11, 'y': 0, 'vx': 0, 'vy': 47.87e3},
    {'name': 'Венера', 'mass': 4.87e24, 'x': 0.72 * 1.496e11, 'y': 0, 'vx': 0, 'vy': 35.02e3},
    {'name': 'Земля', 'mass': 5.97e24, 'x': 1.496e11, 'y': 0, 'vx': 0, 'vy': 29.78e3},
    {'name': 'Марс', 'mass': 6.42e23, 'x': 1.52 * 1.496e11, 'y': 0, 'vx': 0, 'vy': 24.077e3},
]


# Для хранения координат для анимации
trajectory = [[] for _ in range(len(planets))]

# Шаги по времени
num_steps = int(time_duration / dt)

for step in range(num_steps):
    for i, planet in enumerate(planets):
        # Считаем вектор до Солнца
        r_x = -planet['x']
        r_y = -planet['y']
        r = np.sqrt(r_x**2 + r_y**2)

        # Силы
        ax = G * m_sun / r**2 * (r_x / r)
        ay = G * m_sun / r**2 * (r_y / r)

        # Обновление скоростей и позиций
        planet['vx'] += ax * dt
        planet['vy'] += ay * dt
        planet['x'] += planet['vx'] * dt
        planet['y'] += planet['vy'] * dt

        # Сохранение пути
        trajectory[i].append((planet['x'], planet['y']))

# Настройка графики
fig, ax = plt.subplots()
ax.set_xlim(-2 * 1.496e11, 2 * 1.496e11)
ax.set_ylim(-2 * 1.496e11, 2 * 1.496e11)
ax.set_aspect('equal')
ax.set_title('Движение планет вокруг Солнца')

sun = plt.Circle((0, 0), 1.5e10, color='yellow')
ax.add_artist(sun)

# Создаём объекты для планет
planet_lines = [ax.plot([], [], label = p['name'], marker='o')[0] for p in planets]

def init():
    for line in planet_lines:
        line.set_data([], [])
    return planet_lines

def update(frame):
    for i, line in enumerate(planet_lines):
        line.set_data([pos[0] for pos in trajectory[i][:frame+1]], 
                        [pos[1] for pos in trajectory[i][:frame+1]])
    return planet_lines

ax.legend()

# Создаем анимацию
ani = FuncAnimation(fig, update, frames=len(trajectory[0]), init_func=init, blit=True, interval=30)

# Показываем анимацию
plt.show()