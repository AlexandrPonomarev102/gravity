import numpy as np
from scipy.integrate import odeint
import torch
from torchdiffeq import odeint as torch_odeint
import matplotlib.pyplot as plt

# Общие параметры
G = 1.0
masses = [10.0, 1.0, 1.0]
t_max = 100
n_steps = 1000

# =============================================
# Решение с использованием SciPy
# =============================================
def scipy_equations(state, t):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state
    
    # Расчёты расстояний
    r12 = np.sqrt((r2x-r1x)**2 + (r2y-r1y)**2 + 1e-12)
    r13 = np.sqrt((r3x-r1x)**2 + (r3y-r1y)**2 + 1e-12)
    r23 = np.sqrt((r3x-r2x)**2 + (r3y-r2y)**2 + 1e-12)
    
    # Уравнения движения
    dv1xdt = G*(masses[1]*(r2x-r1x))/r12**3 + G*masses[2]*(r3x-r1x)/r13**3
    dv1ydt = G*(masses[1]*(r2y-r1y))/r12**3 + G*masses[2]*(r3y-r1y)/r13**3
    
    dv2xdt = G*masses[0]*(r1x-r2x)/r12**3 + G*masses[2]*(r3x-r2x)/r23**3
    dv2ydt = G*masses[0]*(r1y-r2y)/r12**3 + G*masses[2]*(r3y-r2y)/r23**3
    
    dv3xdt = G*masses[0]*(r1x-r3x)/r13**3 + G*masses[1]*(r2x-r3x)/r23**3
    dv3ydt = G*masses[0]*(r1y-r3y)/r13**3 + G*masses[1]*(r2y-r3y)/r23**3
    
    return [v1x, v1y, dv1xdt, dv1ydt,
            v2x, v2y, dv2xdt, dv2ydt,
            v3x, v3y, dv3xdt, dv3ydt]

# Начальные условия для SciPy
state0_scipy = [0.0, 0.0, 0.5, 0.0,
                2.0, 0.0, 0.0, 2.0,
                -2.0, 0.0, 0.0, -2.0]

# Интегрирование
t_sp = np.linspace(0, t_max, n_steps)
sol_sp = odeint(scipy_equations, state0_scipy, t_sp)

# Расчёт энергий для SciPy
def calculate_energies_scipy(solution):
    KE = np.zeros_like(t_sp)
    PE = np.zeros_like(t_sp)
    
    for i in range(len(t_sp)):
        # Кинетическая энергия
        v1 = np.sqrt(solution[i,2]**2 + solution[i,3]**2)
        v2 = np.sqrt(solution[i,6]**2 + solution[i,7]**2)
        v3 = np.sqrt(solution[i,10]**2 + solution[i,11]**2)
        KE[i] = 0.5*(masses[0]*v1**2 + masses[1]*v2**2 + masses[2]*v3**2)
        
        # Потенциальная энергия
        r12 = np.sqrt((solution[i,4]-solution[i,0])**2 + (solution[i,5]-solution[i,1])**2)
        r13 = np.sqrt((solution[i,8]-solution[i,0])**2 + (solution[i,9]-solution[i,1])**2)
        r23 = np.sqrt((solution[i,8]-solution[i,4])**2 + (solution[i,9]-solution[i,5])**2)
        PE[i] = -G*(masses[0]*masses[1]/r12 + masses[0]*masses[2]/r13 + masses[1]*masses[2]/r23)
    
    return KE, PE, KE + PE

KE_sp, PE_sp, TE_sp = calculate_energies_scipy(sol_sp)

# =============================================
# Решение с использованием PyTorch
# =============================================
def torch_equations(t, state):
    r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y = state.unbind()
    
    # Расчёты расстояний
    r12 = torch.sqrt((r2x-r1x)**2 + (r2y-r1y)**2 + 1e-12)
    r13 = torch.sqrt((r3x-r1x)**2 + (r3y-r1y)**2 + 1e-12)
    r23 = torch.sqrt((r3x-r2x)**2 + (r3y-r2y)**2 + 1e-12)
    
    # Уравнения движения
    dv1xdt = G*(masses[1]*(r2x-r1x))/r12**3 + G*masses[2]*(r3x-r1x)/r13**3
    dv1ydt = G*(masses[1]*(r2y-r1y))/r12**3 + G*masses[2]*(r3y-r1y)/r13**3
    
    dv2xdt = G*masses[0]*(r1x-r2x)/r12**3 + G*masses[2]*(r3x-r2x)/r23**3
    dv2ydt = G*masses[0]*(r1y-r2y)/r12**3 + G*masses[2]*(r3y-r2y)/r23**3
    
    dv3xdt = G*masses[0]*(r1x-r3x)/r13**3 + G*masses[1]*(r2x-r3x)/r23**3
    dv3ydt = G*masses[0]*(r1y-r3y)/r13**3 + G*masses[1]*(r2y-r3y)/r23**3
    
    return torch.stack([v1x, v1y, dv1xdt, dv1ydt,
                        v2x, v2y, dv2xdt, dv2ydt,
                        v3x, v3y, dv3xdt, dv3ydt])

# Начальные условия для PyTorch
state0_torch = torch.tensor(state0_scipy, dtype=torch.float32)
t_torch = torch.linspace(0, t_max, n_steps)

# Интегрирование
sol_torch = torch_odeint(torch_equations, state0_torch, t_torch)

# Расчёт энергий для PyTorch
def calculate_energies_torch(solution):
    v1 = torch.norm(solution[:,2:4], dim=1)
    v2 = torch.norm(solution[:,6:8], dim=1)
    v3 = torch.norm(solution[:,10:12], dim=1)
    
    KE = 0.5*(masses[0]*v1**2 + masses[1]*v2**2 + masses[2]*v3**2)
    
    r1 = solution[:,0:2]
    r2 = solution[:,4:6]
    r3 = solution[:,8:10]
    
    r12 = torch.norm(r2 - r1, dim=1)
    r13 = torch.norm(r3 - r1, dim=1)
    r23 = torch.norm(r3 - r2, dim=1)
    
    PE = -G*(masses[0]*masses[1]/r12 + masses[0]*masses[2]/r13 + masses[1]*masses[2]/r23)
    
    return KE.numpy(), PE.numpy(), (KE + PE).numpy()

KE_torch, PE_torch, TE_torch = calculate_energies_torch(sol_torch)

# =============================================
# Визуализация результатов
# =============================================
# Вычисление абсолютных относительных разностей
# Вычисление полной энергии для обоих методов
TE_sp = KE_sp + PE_sp
TE_torch = KE_torch + PE_torch

# Расчет относительной разницы полной энергии
delta_TE = np.abs((TE_sp - TE_torch) / TE_sp)

# Создание графика
plt.figure(figsize=(14, 8))

# Построение кривой
plt.plot(t_sp, delta_TE, 
         label='Δ(T+P)/|T+P| между методами', 
         linewidth=2, 
         color='purple')


# Настройка оформления
plt.title('Относительная разница полной энергии (T+P)\nмежду методами SciPy и PyTorch', 
          fontsize=14, pad=20)
plt.xlabel('Время', fontsize=12)
plt.ylabel('|SciPy - Torch| / |SciPy|', fontsize=12)
plt.legend(fontsize=12, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')

# Настройка пределов и подписей
plt.ylim(1e-16, 1e-2)
plt.yticks([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
           ['10⁻¹⁶', '10⁻¹⁴', '10⁻¹²', '10⁻¹⁰', '10⁻⁸', '10⁻⁶', '10⁻⁴', '10⁻²'])



plt.tight_layout()
plt.show()