import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve

# Función para calcular los coeficientes según N y b, con tolerancia de error
def compute_coefficients(N: int, b:float, error_tol:float):
    """
    Calcula los coeficientes para una expansión basada en mínimos cuadrados
    dado un número N de términos y un intervalo [0, b], respetando una tolerancia 'tol'.

    Args:
        N (int): Número de términos (orden del modelo).
        b (float): Límite superior del intervalo.
        tol (float): Tolerancia para las integraciones.

    Returns:
        tuple: Coeficientes a, C0 y el arreglo de Cn.
    """
    # Definición de las funciones para integrar
    def I1(beta): return beta**4 * np.cos(beta)
    def I2(beta): return beta**4
    def I3(beta): return beta**2
    def I4(beta, n): return beta**2 * np.cos(n * beta)
    def I5(beta): return beta**2 * np.cos(beta)
    def I6(beta): return 1
    def I7(beta, n): return np.cos(n * beta)
    def I8(beta, m, n): return np.cos(m * beta) * np.cos(n * beta)
    def I9(beta, n): return beta**2 * np.cos(beta) * np.cos(n * beta)

    # Subfunción genérica para integrar con tolerancia personalizada
    def integrate(func, args=()):
        return quad(func, 0, b, args=args, epsabs=error_tol, epsrel=error_tol)[0]

    # Calcular las integrales necesarias
    I1_val = integrate(I1)
    I2_val = integrate(I2)
    I3_val = integrate(I3)
    I5_val = integrate(I5)
    I6_val = integrate(I6)

    # Inicializar la matriz del sistema M y el vector independiente B
    M = np.zeros((N + 2, N + 2))  # Dimensiones dependen de N + 2 ecuaciones
    B = np.zeros(N + 2)           # Vector independiente

    # Llenar las primeras ecuaciones del sistema basadas en mínimos cuadrados
    M[0, 0] = I2_val  # Coeficiente asociado a 'a'
    M[0, 1] = I3_val  # Coeficiente asociado a C0
    for n in range(1, N + 1):
        M[0, 1 + n] = 2 * integrate(I4, (n,))  # Coeficientes asociados a Cn
    B[0] = -2 * I1_val  # Término independiente correspondiente

    M[1, 0] = I3_val  # Segunda ecuación
    M[1, 1] = I6_val  # Coeficiente constante
    for n in range(1, N + 1):
        M[1, 1 + n] = 2 * integrate(I7, (n,))
    B[1] = -2 * I5_val  # Término independiente

    # Ecuaciones adicionales para los coeficientes de orden superior
    for k in range(1, N + 1):
        M[1 + k, 0] = 2 * integrate(I9, (k,))  # Término independiente de a
        M[1 + k, 1] = 2 * integrate(I7, (k,))  # Término para C0
        for n in range(1, N + 1):
            M[1 + k, 1 + n] = 4 * integrate(I8, (k, n))  # Producto cruzado Cn
        B[1 + k] = -4 * integrate(I9, (k,))  # Término independiente

    # Imponer condición de frontera E(0) = 0 en la última ecuación
    M[-1, :] = 0       # Limpiar la última fila
    M[-1, 1] = 1       # Coeficiente de C0
    for n in range(1, N + 1):
        M[-1, 1 + n] = 2  # Coeficiente de Cn
    B[-1] = 0          # Término independiente ajustado

    # Resolver el sistema de ecuaciones lineales
    coefficients = solve(M, B)  # Resolver M * coefficients = B
    a = coefficients[0]        # Coeficiente 'a'
    C0 = coefficients[1]       # Coeficiente C0
    Cn = coefficients[2:]      # Resto de coeficientes Cn

    return a, C0, Cn  # Devolver los coeficientes calculados

# Función para verificar que el error esté dentro de los límites permitidos
def verify_error(a:float, C0:float, Cn:np.ndarray, N:int, b:float, error_tol:float):
    """
    Verifica si el error máximo |E(beta)| está dentro de la tolerancia.

    Args:
        a (float): Coeficiente calculado.
        C0 (float): Primer coeficiente.
        Cn (np.ndarray): Coeficientes restantes.
        N (int): Número de términos.
        b (float): Límite superior del intervalo.
        tol (float): Tolerancia aceptada.

    Returns:
        tuple: (bool, float) donde el bool indica si cumple la tolerancia.
    """
    beta = np.linspace(0, b, 1000)  # Discretizar el intervalo
    E = -beta**2 * (2 * np.cos(beta) + a) - (C0 + 2 * sum(Cn[n] * np.cos((n + 1) * beta) for n in range(N)))
    max_error = np.max(np.abs(E))  # Error máximo absoluto
    return max_error <= error_tol, max_error  # Comparar contra la tolerancia

# Función para encontrar el máximo b que cumple la tolerancia de error
def find_max_b_for_N(N:int, error_tol:float, b_start=0.1, b_step=0.01, max_b=10):
    """
    Encuentra el máximo b para un N fijo que satisfaga la tolerancia.

    Args:
        N (int): Número de términos.
        b_start (float): Valor inicial de b.
        b_step (float): Incremento en b.
        max_b (float): Máximo valor de b a probar.
        error_tol (float): Tolerancia de error.

    Returns:
        tuple: (float, float, tuple) donde están el b máximo, el error y los coeficientes.
    """
    b_current = b_start  # Iniciar desde b_start
    best_b = 0
    best_max_error = 0
    best_coefficients = None

    while b_current <= max_b:
        try:
            # Calcular coeficientes y verificar error
            a, C0, Cn = compute_coefficients(N, b_current, error_tol)
            is_valid, max_error = verify_error(a, C0, Cn, N, b_current, error_tol)
            if is_valid:
                best_b = b_current
                best_max_error = max_error
                best_coefficients = (a, C0, Cn)
                b_current += b_step  # Incrementar b
            else:
                break  # Terminar si no cumple
        except:
            break  # Salir del bucle si ocurre un error

    return best_b, best_max_error, best_coefficients

# Prueba del sistema
N = 2
error_tol = 1e-5

# Buscar el valor máximo de b que satisface la tolerancia especificada
max_b, max_error, coefficients = find_max_b_for_N(N, error_tol)

# Si se encuentra un b válido, se imprimen los resultados y se grafica el error
if max_b > 0:
    # Descomprimir los coeficientes calculados
    a, C0, Cn = coefficients
    # Imprimir los resultados obtenidos
    print(f"¡Solución encontrada para N={N} con b={max_b:.4f}!")
    print(f"Error máximo: {max_error:.2e}")
    print("Coeficientes:")
    print(f"a = {a:.6e}")
    print(f"C0 = {C0:.6f}")
    for n in range(1, N + 1):  # Iterar sobre el resto de los coeficientes
        print(f"C{n} = {Cn[n - 1]:.6f}")
    # Preparar el intervalo para graficar los valores de beta
    beta = np.linspace(0, max_b, 500)
    # Calcular el error E(beta) para cada valor de beta en el intervalo
    E = -beta**2 * (2 * np.cos(beta) + a) - (C0 + 2 * sum(Cn[n] * np.cos((n + 1) * beta) for n in range(N)))
    # Crear la figura para graficar el error
    plt.figure(figsize=(10, 6))
    plt.plot(beta, E, label=f'Error (N={N}, b={max_b:.4f})')
    plt.axhline(y=error_tol, color='r', linestyle='--', label=f'Tolerancia: {error_tol:.0e}')
    plt.axhline(y=-error_tol, color='r', linestyle='--')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$E(\beta)$')
    plt.title(f'Error de dispersión (N={N}, b={max_b:.4f}, Max error={max_error:.2e})')
    plt.legend()
    plt.grid()
    plt.show()
else:
    # Si no se encuentra un b válido, mostrar un mensaje de error
    print(f"No se encontró un b válido para N={N} dentro del rango probado.")