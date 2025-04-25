import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Константы
G = torch.tensor(1.0)
m1 = torch.tensor(10.0)
m2 = torch.tensor(1.0)
m3 = torch.tensor(1.0)

# Функция, описывающая систему уравнений
def equations(t, state):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state.unbind()
    
    # Расчет расстояний между телами
    r12 = torch.sqrt((r2x - r1x)**2 + (r2y - r1y)**2 + 1e-6)
    r13 = torch.sqrt((r3x - r1x)**2 + (r3y - r1y)**2 + 1e-6)
    r23 = torch.sqrt((r3x - r2x)**2 + (r3y - r2y)**2 + 1e-6)
    
    # Правые части уравнений
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
r10 = 0.0   # Центральное тело (x)
r1y0 = 0.0  # Центральное тело (y)
v10 = 0.5   # Центральное тело (vx)
v1y0 = 0.0  # Центральное тело (vy)

r20 = 2.0   # Тело 2 (x)
r2y0 = 0.0  # Тело 2 (y)
v20 = 0.0   # Тело 2 (vx)
v2y0 = 2.0  # Тело 2 (vy)

r30 = -2.0  # Тело 3 (x)
r3y0 = 0.0  # Тело 3 (y)
v30 = 0.0   # Тело 3 (vx)
v3y0 = -2.0 # Тело 3 (vy)

state0 = torch.tensor([
    r10, r1y0, v10, v1y0,
    r20, r2y0, v20, v2y0,
    r30, r3y0, v30, v3y0
], dtype=torch.float32)

# Интервал времени
t = torch.linspace(0, 100, 1000)

# Решение системы
solution = odeint(equations, state0, t)
solution_np = solution.detach().numpy()

# Извлечение координат
p1_x, p1_y = solution_np[:, 0], solution_np[:, 1]
p2_x, p2_y = solution_np[:, 4], solution_np[:, 5]
p3_x, p3_y = solution_np[:, 8], solution_np[:, 9]

# Расчет энергии системы
# Извлечение координат и скоростей
r1_x = solution[:, 0]
r1_y = solution[:, 1]
v1_x = solution[:, 2]
v1_y = solution[:, 3]

r2_x = solution[:, 4]
r2_y = solution[:, 5]
v2_x = solution[:, 6]
v2_y = solution[:, 7]

r3_x = solution[:, 8]
r3_y = solution[:, 9]
v3_x = solution[:, 10]
v3_y = solution[:, 11]

# Кинетическая энергия
ke1 = 0.5 * m1 * (v1_x**2 + v1_y**2)
ke2 = 0.5 * m2 * (v2_x**2 + v2_y**2)
ke3 = 0.5 * m3 * (v3_x**2 + v3_y**2)
kinetic_energy = ke1 + ke2 + ke3

# Потенциальная энергия (с учетом всех пар)
r12 = torch.sqrt((r2_x - r1_x)**2 + (r2_y - r1_y)**2 + 1e-6)
r13 = torch.sqrt((r3_x - r1_x)**2 + (r3_y - r1_y)**2 + 1e-6)
r23 = torch.sqrt((r3_x - r2_x)**2 + (r3_y - r2_y)**2 + 1e-6)

pe12 = -G * m1 * m2 / r12
pe13 = -G * m1 * m3 / r13
pe23 = -G * m2 * m3 / r23
potential_energy = pe12 + pe13 + pe23

# Полная энергия
total_energy = kinetic_energy + potential_energy

# Преобразование в numpy массивы
time = t.detach().numpy()
kinetic_energy_np = kinetic_energy.detach().numpy()
potential_energy_np = potential_energy.detach().numpy()
total_energy_np = total_energy.detach().numpy()

# Построение графика энергии
plt.figure(figsize=(10, 6))
plt.plot(time, kinetic_energy_np, label='Кинетическая энергия')
plt.plot(time, potential_energy_np, label='Потенциальная энергия')
plt.plot(time, total_energy_np, label='Полная энергия', linestyle='--')
plt.xlabel('Время')
plt.ylabel('Энергия')
plt.title('Энергия системы трех тел')
plt.legend()
plt.grid(True)
plt.savefig('three_body_energy.png')
plt.show()