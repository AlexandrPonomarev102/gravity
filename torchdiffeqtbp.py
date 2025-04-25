import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import psutil
import os
import time  # Импортируем модуль для замера времени

# Функция для получения используемой памяти в МБ
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

# Константы
G = torch.tensor(1.0)
m1 = torch.tensor(10.0)
m2 = torch.tensor(1.0)
m3 = torch.tensor(1.0)

# Вывод начального использования памяти
print(f"[{time.strftime('%H:%M:%S')}] Начальное использование памяти: {get_memory_usage():.2f} МБ")

# Функция, описывающая систему уравнений
def equations(t, state):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state.unbind()
    
    r12 = torch.sqrt((r2x - r1x)**2 + (r2y - r1y)**2 + 1e-6)
    r13 = torch.sqrt((r3x - r1x)**2 + (r3y - r1y)**2 + 1e-6)
    r23 = torch.sqrt((r3x - r2x)**2 + (r3y - r2y)**2 + 1e-6)
    
    dv1xdt = G * (m2 * (r2x - r1x) / r12**3 + m3 * (r3x - r1x) / r13**3)
    dv1ydt = G * (m2 * (r2y - r1y) / r12**3 + m3 * (r3y - r1y) / r13**3)
    
    dv2xdt = G * (m1 * (r1x - r2x) / r12**3 + m3 * (r3x - r2x) / r23**3)
    dv2ydt = G * (m1 * (r1y - r2y) / r12**3 + m3 * (r3y - r2y) / r23**3)
    
    dv3xdt = G * (m1 * (r1x - r3x) / r13**3 + m2 * (r2x - r3x) / r23**3)
    dv3ydt = G * (m1 * (r1y - r3y) / r13**3 + m2 * (r2y - r3y) / r23**3)
    
    return torch.stack([
        v1x, v1y, dv1xdt, dv1ydt,
        v2x, v2y, dv2xdt, dv2ydt,
        v3x, v3y, dv3xdt, dv3ydt
    ])

# Начальные условия
state0 = torch.tensor([
    0.0, 0.0, 0.5, 0.0,
    2.0, 0.0, 0.0, 2.0,
    -2.0, 0.0, 0.0, -2.0
], dtype=torch.float32)

# Интервал времени
t = torch.linspace(0, 100, 1000)

# Решение системы
print(f"[{time.strftime('%H:%M:%S')}] Начало решения ODE")
start_ode = time.time()
solution = odeint(equations, state0, t)
print(f"[{time.strftime('%H:%M:%S')}] Решение ODE завершено за {time.time()-start_ode:.2f} сек")
print(f"Память после решения ODE: {get_memory_usage():.2f} МБ")
# Конвертация в numpy array
solution_np = solution.detach().numpy()

# Извлечение координат
p1_x, p1_y = solution_np[:, 0], solution_np[:, 1]
p2_x, p2_y = solution_np[:, 4], solution_np[:, 5]
p3_x, p3_y = solution_np[:, 8], solution_np[:, 9]

# Создание анимации
fig, ax = plt.subplots(figsize=(8, 8))
writer = animation.PillowWriter(fps=30)

print(f"\n[{time.strftime('%H:%M:%S')}] Начало создания анимации")
start_animation = time.time()

with writer.saving(fig, "three_body_problem_torch.gif", 100):
    for i in range(len(p1_x)):
        ax.clear()
        ax.set_xlim(-2.5, 20)
        ax.set_ylim(-10, 10)
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        
        ax.plot(p1_x[:i+1], p1_y[:i+1], color="blue", alpha=0.5, label="Central")
        ax.plot(p2_x[:i+1], p2_y[:i+1], color="green", alpha=0.5, label="Body 2")
        ax.plot(p3_x[:i+1], p3_y[:i+1], color="red", alpha=0.5, label="Body 3")
        
        ax.plot(p1_x[i], p1_y[i], 'o', color="blue")
        ax.plot(p2_x[i], p2_y[i], 'o', color="green")
        ax.plot(p3_x[i], p3_y[i], 'o', color="red")
        
        ax.legend()
        writer.grab_frame()

animation_time = time.time() - start_animation
print(f"[{time.strftime('%H:%M:%S')}] Анимация создана за {animation_time:.2f} сек")

# Итоговая информация
print(f"\nИтоги:")
print(f"Общее время выполнения: {time.time()-start_ode:.2f} сек")
print(f"Пиковое использование памяти: {get_memory_usage():.2f} МБ")
print("Анимация сохранена как 'three_body_problem_torch.gif'")