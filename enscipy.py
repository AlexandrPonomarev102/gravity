import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Константы
G = 1.0  # Гравитационная постоянная
m1 = 10.0  # Масса центрального тела
m2 = 1.0  # Масса первого вращающегося тела
m3 = 1.0  # Масса второго вращающегося тела

# Функция, описывающая систему уравнений
def equations(state, t):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state
    
    # Расчет расстояний между телами
    r12 = np.sqrt((r2x - r1x)**2 + (r2y - r1y)**2)
    r13 = np.sqrt((r3x - r1x)**2 + (r3y - r1y)**2)
    r23 = np.sqrt((r3x - r2x)**2 + (r3y - r2y)**2)
    
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
r10, r1y0, v10, v1y0 = 0.0, 0.0, 0.5, 0.0
r20, r2y0, v20, v2y0 = 2.0, 0.0, 0.0, 2.0
r30, r3y0, v30, v3y0 = -2.0, 0.0, 0.0, -2.0

state0 = [r10, r1y0, v10, v1y0, r20, r2y0, v20, v2y0, r30, r3y0, v30, v3y0]

# Интервал времени
t = np.linspace(0, 100, 1000)

# Решение системы уравнений
solution = odeint(equations, state0, t)

# Функция для расчета кинетической энергии
def kinetic_energy(state):
    v1x, v1y = state[2], state[3]
    v2x, v2y = state[6], state[7]
    v3x, v3y = state[10], state[11]
    return 0.5 * (m1 * (v1x**2 + v1y**2) + m2 * (v2x**2 + v2y**2) + m3 * (v3x**2 + v3y**2))

# Функция для расчета потенциальной энергии
def potential_energy(state):
    r1x, r1y = state[0], state[1]
    r2x, r2y = state[4], state[5]
    r3x, r3y = state[8], state[9]
    r12 = np.sqrt((r2x - r1x)**2 + (r2y - r1y)**2)
    r13 = np.sqrt((r3x - r1x)**2 + (r3y - r1y)**2)
    r23 = np.sqrt((r3x - r2x)**2 + (r3y - r2y)**2)
    return -G * (m1*m2/r12 + m1*m3/r13 + m2*m3/r23)

# Расчет энергий
KE = np.array([kinetic_energy(state) for state in solution])
PE = np.array([potential_energy(state) for state in solution])
TE = KE + PE

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(t, KE, label='Кинетическая энергия')
plt.plot(t, PE, label='Потенциальная энергия')
plt.plot(t, TE, label='Полная энергия')
plt.xlabel('Время')
plt.ylabel('Энергия')
plt.title('Энергии в системе трех тел')
plt.legend()
plt.grid(True)
plt.show()
