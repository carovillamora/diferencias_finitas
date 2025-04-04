import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve

# Definir la función principal para calcular coeficientes imponiendo E(0) = 0
def compute_coefficients(N:int, b:float):
    """
    Calcula los coeficientes de un sistema ajustado mediante mínimos cuadrados,
    asegurando que se cumpla la condición de frontera E(0) = 0.
    
    Parámetros:
    -----------
        N (int): Número de términos en la expansión de coeficientes (orden del modelo).
        b (float): Límite superior del intervalo de integración.
    
    Retorna:
    --------
        a (float): Coeficiente independiente.
        C0 (float): Coeficiente inicial.
        Cn (np.ndarray): Arreglo con los coeficientes restantes.
    """
    # Definición de las funciones para calcular las integrales
    def I1(beta): return beta**4 * np.cos(beta)  # Para el término beta^4*cos(beta)
    def I2(beta): return beta**4                # Para el término beta^4
    def I3(beta): return beta**2                # Para el término beta^2
    def I4(beta, n): return beta**2 * np.cos(n * beta)  # Beta^2*cos(n*beta)
    def I5(beta): return beta**2 * np.cos(beta)         # Beta^2*cos(beta)
    def I6(beta): return 1                              # Constante para integración
    def I7(beta, n): return np.cos(n * beta)            # Cos(n*beta)
    def I8(beta, m, n): return np.cos(m * beta) * np.cos(n * beta)  # Producto de cosenos
    def I9(beta, n): return beta**2 * np.cos(beta) * np.cos(n * beta)  # Beta^2*cos(beta)*cos(n*beta)

    # Calcular las integrales en el intervalo [0, b]
    # Se retorna tanto el valor de la integral como el error, que ignoramos (_) aquí.
    I1_val, _ = quad(I1, 0, b)
    I2_val, _ = quad(I2, 0, b)
    I3_val, _ = quad(I3, 0, b)
    I5_val, _ = quad(I5, 0, b)
    I6_val, _ = quad(I6, 0, b)

    # Inicializar la matriz M y el vector B para el sistema de ecuaciones
    M_size = N + 2  # Tamaño de la matriz depende de N + 2 ecuaciones
    M = np.zeros((M_size, M_size))  # Matriz de coeficientes vacía
    B = np.zeros(M_size)            # Vector independiente vacío

    # Llenar las primeras N+1 ecuaciones basadas en mínimos cuadrados
    # Primera ecuación para el término inicial
    M[0, 0] = I2_val
    M[0, 1] = I3_val
    for n in range(1, N + 1):
        M[0, 1 + n] = 2 * quad(lambda beta: I4(beta, n), 0, b)[0]  # Coeficientes para Cn
    B[0] = -2 * I1_val  # Término independiente correspondiente

    # Segunda ecuación para otro ajuste
    M[1, 0] = I3_val
    M[1, 1] = I6_val
    for n in range(1, N + 1):
        M[1, 1 + n] = 2 * quad(lambda beta: I7(beta, n), 0, b)[0]  # Coeficientes para Cn
    B[1] = -2 * I5_val  # Término independiente

    # Resto de las ecuaciones para los coeficientes de orden superior
    for k in range(1, N + 1):
        M[1 + k, 0] = 2 * quad(lambda beta: I9(beta, k), 0, b)[0]
        M[1 + k, 1] = 2 * quad(lambda beta: I7(beta, k), 0, b)[0]
        for n in range(1, N + 1):
            M[1 + k, 1 + n] = 4 * quad(lambda beta: I8(beta, k, n), 0, b)[0]  # Producto cruzado
        B[1 + k] = -4 * quad(lambda beta: I9(beta, k), 0, b)[0]

    # Imponer explícitamente la condición E(0) = 0
    # Se reemplaza la última fila de la matriz por esta condición
    M[-1, :] = 0       # Limpiar la fila para evitar conflicto
    M[-1, 1] = 1       # Coeficiente de C0
    for n in range(1, N + 1):
        M[-1, 1 + n] = 2  # Coeficiente para Cn
    B[-1] = 0          # Término independiente ajustado a E(0) = 0

    # Resolver el sistema de ecuaciones lineales
    coefficients = solve(M, B)  # Resolver M * coefficients = B
    a = coefficients[0]        # Coeficiente "a"
    C0 = coefficients[1]       # Coeficiente C0
    Cn = coefficients[2:]      # Resto de coeficientes Cn

    return a, C0, Cn  # Retornar los coeficientes calculados

# Parámetros iniciales para la prueba
N = 2  # Orden del modelo
b = 0.33  # Límite superior del intervalo

# Calcular los coeficientes imponiendo la condición E(0) = 0
a, C0, Cn = compute_coefficients(N, b)
print(f"Coeficientes para N={N}:")
print(f"a = {a:.6e}")  # Mostrar coeficiente "a" con precisión científica
print(f"C0 = {C0:.6f}")  # Mostrar coeficiente C0 con 6 decimales
for n in range(1, N + 1):
    print(f"C{n} = {Cn[n - 1]:.6f}")  # Mostrar los coeficientes Cn con precisión

# Graficar el error usando los coeficientes calculados
beta = np.linspace(0, b, 500)  # Valores de beta en el intervalo [0, b]
E = -beta**2 * (2 * np.cos(beta) + a) - (C0 + 2 * sum(Cn[n] * np.cos((n + 1) * beta) for n in range(N)))

plt.figure(figsize=(10, 6))
plt.plot(beta, E, label=f'Error (N={N})')  # Graficar el error
plt.axhline(1e-5, color='red', linestyle='--')  # Líneas de referencia para tolerancia
plt.axhline(-1e-5, color='red', linestyle='--')
plt.xlabel(r'$\beta$')  # Etiqueta en el eje x con símbolo matemático
plt.ylabel(r'$E(\beta)$')  # Etiqueta en el eje y con símbolo matemático
plt.title('Error de dispersión')  # Título del gráfico
plt.legend()  # Agregar leyenda
plt.grid()  # Mostrar cuadrícula en el gráfico
plt.show()