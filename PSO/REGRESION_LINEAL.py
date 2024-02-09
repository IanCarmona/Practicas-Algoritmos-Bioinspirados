# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from pyswarm import pso

# # Leer datos desde el archivo CSV
# data = pd.read_csv("Salary.csv")
# X = data['YearsExperience'].values.reshape(-1, 1)
# y = data['Salary'].values

# # Definir la función de error para la regresión lineal
# def error_function(params):
#     a, b = params
#     y_pred = a * X + b
#     error = np.mean((y_pred - y) ** 2)
#     return error

# # Definir los límites para los parámetros 'a' y 'b'
# lb = [-10000, -10000]
# ub = [10000, 10000]


# poblacion = 10
# iteraciones = 100
# w = 0.5
# c1 = 1
# c2 = 2


# # Ejecutar la optimización de enjambre de partículas (PSO)
# best_params, _ = pso(error_function, lb, ub, swarmsize=poblacion, maxiter=iteraciones, omega=w, phig=c2, phip=c1)


# # Recuperar los parámetros óptimos
# a_optimal, b_optimal = best_params

# # Crear el modelo de regresión lineal con los parámetros óptimos
# regression_model = LinearRegression()
# regression_model.fit(X, y)

# # Hacer predicciones
# y_pred = regression_model.predict(X)

# # Graficar los datos y la línea de regresión
# plt.scatter(X, y, label='Datos reales', color='green')
# plt.plot(X, y_pred, color='red', linewidth=1, label='Regresión lineal')
# plt.xlabel('Años de experiencia')
# plt.ylabel('Salario')
# plt.legend()
# plt.title('Regresión Lineal con PSO')
# plt.show()


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

random.seed(7)

# Función de regresión lineal
def linear_regression(a, b, x):
    return a + b * x

# Función para calcular el error cuadrático medio
def mean_squared_error(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# Función de fitness para el PSO (error cuadrático medio)
def fitness_function(position):
    a, b = position
    y_hat = linear_regression(a, b, X)
    error = mean_squared_error(y, y_hat)
    return error

# Carga los datos desde el archivo CSV
data = pd.read_csv('Salary.csv')
X = data['YearsExperience'].values
y = data['Salary'].values

# Configura los parámetros del PSO
num_particles = 200
num_dimensions = 2
max_iterations = 1000
c1 = 1
c2 = 2
w = 0.25

# Inicializa las partículas en un rango razonable para 'a' y 'b'
a_initial = np.random.uniform(min(X), max(X))
b_initial = np.random.uniform(min(y), max(y))
particles = np.column_stack((a_initial * np.ones(num_particles), b_initial * np.ones(num_particles)))

velocities = np.random.rand(num_particles, num_dimensions)

# Inicializa las mejores posiciones locales y globales
best_local_positions = particles.copy()
best_local_errors = np.zeros(num_particles)

best_global_position = None
best_global_error = float('inf')

# Ejecuta el algoritmo PSO
for iteration in range(max_iterations):
    for i in range(num_particles):
        error = fitness_function(particles[i])

        if error < best_local_errors[i]:
            best_local_positions[i] = particles[i]
            best_local_errors[i] = error

        if error < best_global_error:
            best_global_position = particles[i]
            best_global_error = error

    for i in range(num_particles):
        r1, r2 = random.uniform(0, 1), random.uniform(0, 1)
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (best_local_positions[i] - particles[i]) +
                         c2 * r2 * (best_global_position - particles[i]))
        particles[i] += velocities[i]

# Obtiene los valores óptimos encontrados para 'a' y 'b'
optimal_a, optimal_b = best_global_position
print(f"Valores óptimos encontrados: a = {optimal_a}, b = {optimal_b}")

# Realiza la regresión lineal final
predicted_y = linear_regression(optimal_a, optimal_b, X)

# Calcula el error cuadrático medio final
final_error = mean_squared_error(y, predicted_y)
print(f"Error cuadrático medio final: {final_error}")

# Gráfica de los datos originales
plt.scatter(X, y, label='Datos Originales', color='green')

# Gráfica de la regresión lineal
x_range = np.linspace(min(X), max(X), 100)
y_range = linear_regression(optimal_a, optimal_b, x_range)
plt.plot(x_range, y_range, color='red', label='Regresión Lineal')

# Ajustar los límites de los ejes
plt.xlim(min(X), max(X))
plt.ylim(min(y), max(y))

plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.title('Regresión Lineal con PSO')
plt.legend()
plt.show()
