import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import psutil
import os
import time  # Добавляем модуль для работы со временем

# Функция для получения используемой памяти в МБ
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # в МБ

# Константы
G = 1.0  # Гравитационная постоянная
m1 = 10.0  # Масса центрального тела
m2 = 1.0  # Масса первого вращающегося тела
m3 = 1.0  # Масса второго вращающегося тела

# Начальные замеры
start_time = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Начальное использование памяти: {get_memory_usage():.2f} МБ")

# Функция, описывающая систему уравнений
def equations(state, t):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state
    
    # Расчет расстояний между телами
    r12 = np.sqrt((r2x - r1x)**2 + (r2y - r1y)**2 + 1e-12)
    r13 = np.sqrt((r3x - r1x)**2 + (r3y - r1y)**2 + 1e-12)
    r23 = np.sqrt((r3x - r2x)**2 + (r3y - r2y)**2 + 1e-12)
    
    # Правые части уравнений
    dv1xdt = G * (m2 * (r2x - r1x) / r12**3 + m3 * (r3x - r1x) / r13**3)
    dv1ydt = G * (m2 * (r2y - r1y) / r12**3 + m3 * (r3y - r1y) / r13**3)
    
    dv2xdt = G * (m1 * (r1x - r2x) / r12**3 + m3 * (r3x - r2x) / r23**3)
    dv2ydt = G * (m1 * (r1y - r2y) / r12**3 + m3 * (r3y - r2y) / r23**3)
    
    dv3xdt = G * (m1 * (r1x - r3x) / r13**3 + m2 * (r2x - r3x) / r23**3)
    dv3ydt = G * (m1 * (r1y - r3y) / r13**3 + m2 * (r2y - r3y) / r23**3)
    
    # Система уравнений первого порядка
    return [v1x, v1y, dv1xdt, dv1ydt, v2x, v2y, dv2xdt, dv2ydt, v3x, v3y, dv3xdt, dv3ydt]

# Начальные условия
state0 = [
    0.0, 0.0, 0.5, 0.0,   # Центральное тело
    2.0, 0.0, 0.0, 2.0,   # Тело 2
    -2.0, 0.0, 0.0, -2.0  # Тело 3
]

# Интервал времени
t = np.linspace(0, 100, 1000)

# Решение системы уравнений
print(f"\n[{time.strftime('%H:%M:%S')}] Начало решения ODE")
ode_start = time.time()
solution = odeint(equations, state0, t)
ode_time = time.time() - ode_start
print(f"[{time.strftime('%H:%M:%S')}] Решение ODE завершено за {ode_time:.2f} сек")
print(f"Память после решения ODE: {get_memory_usage():.2f} МБ")

# Создание фигуры
fig, ax = plt.subplots(figsize=(8, 8))

# Извлечение координат для каждого тела
p1_x, p1_y = solution[:, 0], solution[:, 1]
p2_x, p2_y = solution[:, 4], solution[:, 5]
p3_x, p3_y = solution[:, 8], solution[:, 9]

# Настройка писателя для сохранения GIF
writer = animation.PillowWriter(fps=30)

# Создание анимации
print(f"\n[{time.strftime('%H:%M:%S')}] Начало создания анимации")
animation_start = time.time()

with writer.saving(fig, "three_body_problem.gif", 100):
    for i in range(len(p1_x)):
        ax.clear()
        ax.set_xlim(-2.5, 20)
        ax.set_ylim(-10, 10)
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        
        ax.set_xlabel("meters")
        ax.set_ylabel("meters")
        
        # Рисуем траектории
        ax.plot(p1_x[:i+1], p1_y[:i+1], color="blue", alpha=0.5, label="p1")
        ax.plot(p2_x[:i+1], p2_y[:i+1], color="green", alpha=0.5, label="p2")
        ax.plot(p3_x[:i+1], p3_y[:i+1], color="red", alpha=0.5, label="p3")
        
        # Рисуем текущие позиции тел
        ax.plot(p1_x[i], p1_y[i], marker=".", color="blue")
        ax.plot(p2_x[i], p2_y[i], marker=".", color="green")
        ax.plot(p3_x[i], p3_y[i], marker=".", color="red")
        
        ax.legend()
        writer.grab_frame()

animation_time = time.time() - animation_start
total_time = time.time() - start_time

# Итоговый отчет
print(f"\n[{time.strftime('%H:%M:%S')}] Анимация создана за {animation_time:.2f} сек")
print(f"Пиковое использование памяти: {get_memory_usage():.2f} МБ")
print(f"\nИтоговое время выполнения: {total_time:.2f} сек")
print("Анимация сохранена как 'three_body_problem.gif'")