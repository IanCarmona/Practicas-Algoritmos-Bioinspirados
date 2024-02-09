
import re
import statistics

# Nombre del archivo de texto que contiene los datos
archivo_txt = "combinacion_0.5_2_2.txt"
# Listas para almacenar los valores de "Best Fitness Value"
best_fitness_values = []

# Leer el archivo y extraer los valores de "Best Fitness Value"
with open(archivo_txt, "r") as archivo:
    lines = archivo.readlines()
    for line in lines:
        match = re.search(r"Best Fitness Value: ([0-9.-]+)", line)
        if match:
            best_fitness_values.append(float(match.group(1)))

# Calcular mínimo, máximo, promedio y desviación estándar
min_fitness = min(best_fitness_values)
max_fitness = max(best_fitness_values)
avg_fitness = statistics.mean(best_fitness_values)
std_deviation = statistics.stdev(best_fitness_values)

# Imprimir resultados
print(f"Valor Mínimo de Best Fitness Value: {min_fitness}")
print(f"Valor Máximo de Best Fitness Value: {max_fitness}")
print(f"Promedio de Best Fitness Value: {avg_fitness}")
print(f"Desviación Estándar de Best Fitness Value: {std_deviation}")

