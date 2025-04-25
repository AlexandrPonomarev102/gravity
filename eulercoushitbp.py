import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметры системы
G = 1.0
m1 = 10.0
m2 = 1.0
m3 = 1.0
t_max = 100
n_steps = 1000
dt = t_max / n_steps

def compute_accelerations(state):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state
    
    # Расстояния между телами
    r12 = np.sqrt((r2x - r1x)**2 + (r2y - r1y)**2 + 1e-12)
    r13 = np.sqrt((r3x - r1x)**2 + (r3y - r1y)**2 + 1e-12)
    r23 = np.sqrt((r3x - r2x)**2 + (r3y - r2y)**2 + 1e-12)
    
    # Ускорения
    a1x = G*(m2*(r2x-r1x)/r12**3 + m3*(r3x-r1x)/r13**3)
    a1y = G*(m2*(r2y-r1y)/r12**3 + m3*(r3y-r1y)/r13**3)
    
    a2x = G*m1*(r1x-r2x)/r12**3 + G*m3*(r3x-r2x)/r23**3
    a2y = G*m1*(r1y-r2y)/r12**3 + G*m3*(r3y-r2y)/r23**3
    
    a3x = G*m1*(r1x-r3x)/r13**3 + G*m2*(r2x-r3x)/r23**3
    a3y = G*m1*(r1y-r3y)/r13**3 + G*m2*(r2y-r3y)/r23**3
    
    return np.array([
        v1x, v1y, a1x, a1y,  # Тело 1
        v2x, v2y, a2x, a2y,  # Тело 2
        v3x, v3y, a3x, a3y   # Тело 3
    ])

# Начальные условия
state = np.array([
    0.0, 0.0, 0.5, 0.0,   # Тело 1 (x, y, vx, vy)
    2.0, 0.0, 0.0, 2.0,   # Тело 2
    -2.0, 0.0, 0.0, -2.0  # Тело 3
])

# Массивы для хранения результатов
positions = np.zeros((n_steps, 12))
energies = np.zeros((n_steps, 3))  # KE, PE, TE

# Метод Эйлера-Коши
for i in range(n_steps):
    positions[i] = state
    
    # Шаг 1: Вычисление промежуточного состояния (предиктор Эйлера)
    k1 = compute_accelerations(state)
    state_pred = state + dt * k1
    
    # Шаг 2: Вычисление корректора
    k2 = compute_accelerations(state_pred)
    state = state + 0.5 * dt * (k1 + k2)
    
    # Расчет энергий
    r12 = np.sqrt((state[4]-state[0])**2 + (state[5]-state[1])**2)
    r13 = np.sqrt((state[8]-state[0])**2 + (state[9]-state[1])**2)
    r23 = np.sqrt((state[8]-state[4])**2 + (state[9]-state[5])**2)
    
    v1 = np.sqrt(state[2]**2 + state[3]**2)
    v2 = np.sqrt(state[6]**2 + state[7]**2)
    v3 = np.sqrt(state[10]**2 + state[11]**2)
    
    KE = 0.5*(m1*v1**2 + m2*v2**2 + m3*v3**2)
    PE = -G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
    
    energies[i] = [KE, PE, KE + PE]

# Анимация (остается без изменений)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

p1_x, p1_y = positions[:,0], positions[:,1]
p2_x, p2_y = positions[:,4], positions[:,5]
p3_x, p3_y = positions[:,8], positions[:,9]

writer = animation.PillowWriter(fps=30)
with writer.saving(fig, "euler_cauchy_three_body.gif", 100):
    for i in range(n_steps):
        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"Time: {i*dt:.1f}")
        
        ax.plot(p1_x[:i+1], p1_y[:i+1], color="blue", alpha=0.3)
        ax.plot(p2_x[:i+1], p2_y[:i+1], color="green", alpha=0.3)
        ax.plot(p3_x[:i+1], p3_y[:i+1], color="red", alpha=0.3)
        
        ax.plot(p1_x[i], p1_y[i], 'o', color="blue", markersize=10)
        ax.plot(p2_x[i], p2_y[i], 'o', color="green", markersize=6)
        ax.plot(p3_x[i], p3_y[i], 'o', color="red", markersize=6)
        
        writer.grab_frame()

# График энергий
t = np.linspace(0, t_max, n_steps)
plt.figure(figsize=(10, 6))
plt.plot(t, energies[:,0], color='blue', label='Кинетическая')
plt.plot(t, energies[:,1], color='orange', label='Потенциальная')
plt.plot(t, energies[:,2], color='green', label='Полная')

plt.title('Энергии системы (метод Эйлера-Коши)')
plt.xlabel('Время')
plt.ylabel('Энергия')
plt.legend()
plt.grid(True)
plt.savefig('euler_cauchy_energies.png')
plt.show()