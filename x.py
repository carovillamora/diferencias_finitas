import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve

# --- Método de mínimos cuadrados (LSM) --- (Extraído de calculo_b.py)
def compute_coefficients_LSM(N, b, error_tol):
    """Calcula coeficientes usando mínimos cuadrados."""
    def I1(beta): return beta**4 * np.cos(beta)
    def I2(beta): return beta**4
    def I3(beta): return beta**2
    def I4(beta, n): return beta**2 * np.cos(n * beta)
    def I5(beta): return beta**2 * np.cos(beta)
    def I6(beta): return 1
    def I7(beta, n): return np.cos(n * beta)
    def I8(beta, m, n): return np.cos(m * beta) * np.cos(n * beta)
    def I9(beta, n): return beta**2 * np.cos(beta) * np.cos(n * beta)

    def integrate(func, args=()):
        return quad(func, 0, b, args=args, epsabs=error_tol, epsrel=error_tol)[0]

    # Integrales
    I1_val = integrate(I1)
    I2_val = integrate(I2)
    I3_val = integrate(I3)
    I5_val = integrate(I5)
    I6_val = integrate(I6)

    # Sistema de ecuaciones
    M = np.zeros((N + 2, N + 2))
    B = np.zeros(N + 2)
    
    M[0, 0] = I2_val
    M[0, 1] = I3_val
    for n in range(1, N + 1):
        M[0, 1 + n] = 2 * integrate(I4, (n,))
    B[0] = -2 * I1_val

    M[1, 0] = I3_val
    M[1, 1] = I6_val
    for n in range(1, N + 1):
        M[1, 1 + n] = 2 * integrate(I7, (n,))
    B[1] = -2 * I5_val

    for k in range(1, N + 1):
        M[1 + k, 0] = 2 * integrate(I9, (k,))
        M[1 + k, 1] = 2 * integrate(I7, (k,))
        for n in range(1, N + 1):
            M[1 + k, 1 + n] = 4 * integrate(I8, (k, n))
        B[1 + k] = -4 * integrate(I9, (k,))

    M[-1, :] = 0
    M[-1, 1] = 1
    for n in range(1, N + 1):
        M[-1, 1 + n] = 2
    B[-1] = 0

    coefficients = solve(M, B)
    a, C0, Cn = coefficients[0], coefficients[1], coefficients[2:]
    return a, C0, Cn

def error_LSM(a, C0, Cn, beta):
    """Calcula el error para el LSM."""
    N = len(Cn)
    return -beta**2 * (2 * np.cos(beta) + a) - (C0 + 2 * sum(Cn[n] * np.cos((n + 1) * beta) for n in range(N)))

# --- Método de series de Taylor (TEM) ---
def coefficients_TEM(N):
    """Calcula coeficientes usando series de Taylor."""
    c = np.zeros(N + 1)
    for m in range(1, N + 1):
        c[m] = ((-1)**(m + 1)) / (m**2) * np.prod([n**2 / (n**2 - m**2) for n in range(1, N + 1) if n != m])
    c[0] = -2 * np.sum(c[1:])
    return c

def error_TEM(c, beta):
    """Calcula el error para el TEM."""
    return sum(c[m] * (2 - 2 * np.cos(m * beta)) for m in range(len(c))) - beta**2

# --- Parámetros de comparación ---
N = 2  # Número de términos
error_tol = 1e-5  # Tolerancia para LSM
b_max_LSM = 0.33  # Valor máximo de b para LSM (ejemplo del artículo)

# Calcular coeficientes
a, C0, Cn = compute_coefficients_LSM(N, b_max_LSM, error_tol)
c_TEM = coefficients_TEM(N)

# Rango de beta para graficar
beta = np.linspace(0, b_max_LSM, 500)

# Calcular errores
E_LSM = error_LSM(a, C0, Cn, beta)
E_TEM = error_TEM(c_TEM, beta)

# --- Gráficos ---
plt.figure(figsize=(12, 6))

# Gráfico de errores
plt.plot(beta, E_LSM, label=f'LSM (N={N}, b={b_max_LSM:.2f})', color='blue')
plt.plot(beta, E_TEM, label=f'TEM (N={N})', color='red', linestyle='--')
plt.axhline(y=error_tol, color='green', linestyle=':', label=f'Tolerancia LSM: {error_tol:.0e}')
plt.axhline(y=-error_tol, color='green', linestyle=':')
plt.xlabel(r'$\beta$ (Número de onda normalizado)')
plt.ylabel(r'Error $E(\beta)$')
plt.title('Comparación de errores: LSM vs. TEM')
plt.legend()
plt.grid()

# Gráfico de coeficientes
plt.figure(figsize=(10, 4))
plt.bar(np.arange(len(Cn)) + 1, Cn, color='blue', alpha=0.6, label='LSM (Cn)')
plt.bar(np.arange(len(c_TEM)), c_TEM, color='red', alpha=0.6, label='TEM (cm)')
plt.xlabel('Índice del coeficiente')
plt.ylabel('Valor del coeficiente')
plt.title('Comparación de coeficientes: LSM vs. TEM')
plt.legend()
plt.grid(axis='y')

plt.show()

# --- Resultados numéricos ---
print("\n--- Coeficientes LSM ---")
print(f"a = {a:.6e}")
print(f"C0 = {C0:.6f}")
for n in range(len(Cn)):
    print(f"C{n+1} = {Cn[n]:.6f}")

print("\n--- Coeficientes TEM ---")
for m in range(len(c_TEM)):
    print(f"c{m} = {c_TEM[m]:.6f}")

print("\n--- Comparación de rangos ---")
print(f"LSM: Máximo b con error < {error_tol:.0e}: {b_max_LSM:.2f}")
print("TEM: Precisión alta solo cerca de β ≈ 0 (no hay b máximo controlado).")