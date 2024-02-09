# simulacion PSO
import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt


# primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
primos = [71]

X = []
Y = []

# rango = [-5.12, 5.12] # RASTRIGIN  


# def fitness_function(*args): # RASTRIGIN 
#     A = 10
#     sum_term = sum(x**2 - A * np.cos(2 * np.pi * x) for x in args)
#     return A * len(args) + sum_term


# rango = [-32.768, 32.768] # ACKLEY  "ALDO"

# def fitness_function(*args):  # ACKLEY
#     A = 20
#     B = 0.2
#     C = 2 * math.pi
    
#     sum_term1 = -A * np.exp(-B * np.sqrt(0.5 * sum(x**2 for x in args)))
#     sum_term2 = -np.exp(0.5 * sum(math.cos(C * x) for x in args))
    
#     result = sum_term1 + sum_term2 + A + math.exp(1)
#     return result


rango = [-2.048, 2.048] # Rosenbrock    

def fitness_function(*args):  # Rosenbrock 
    n = len(args)
    sum_term = sum((1 - args[i])**2 + 100 * (args[i + 1] - args[i]**2)**2 for i in range(n - 1))
    return sum_term




# rango = [-600, 600] # Grewank  


# def fitness_function(*args):  # Grewank 
#     n = len(args)
#     sum_term1 = sum(x**2 / 4000 for x in args)
#     sum_term2 = np.prod([math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(args)])
#     return sum_term1 - sum_term2 + 1


def update_velocity(particle, velocity, pbest, gbest, w, c1, c2, max=1.0):
    # Initialise new velocity array
    num_particle = len(particle)
    new_velocity = np.array([0.0 for i in range(num_particle)])
    # Randomly generate r1, r2 and inertia weight from normal distribution
    r1 = random.uniform(0, max)
    r2 = random.uniform(0, max)
    # Calculate new velocity for all dimensions
    for i in range(num_particle):
        new_velocity[i] = (
            w * velocity[i]
            + c1 * r1 * (pbest[i] - particle[i])
            + c2 * r2 * (gbest[i] - particle[i])
        )
    return new_velocity

def grafica(gbest):
    y = fitness_function(*gbest)  # Desempaqueta gbest
    return y

#para que no sesalga de los límites
def update_position(particle, velocity, rango):
    # Move particles by adding velocity
    new_particle = particle + velocity

    # Asegúrate de que cada componente de la partícula esté dentro del rango
    for i in range(len(new_particle)):
        if new_particle[i] < rango[0]:
            new_particle[i] = rango[0]
        if new_particle[i] > rango[1]:
            new_particle[i] = rango[1]
    
    return new_particle


def pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion, c1, c2, w):
    # Initialisation
    
    resultados = {
        "Global Best Position": None,
        "Best Fitness Value": None,
        "Average Particle Best Fitness Value": None,
        "Number of Generation": None,
    }
    
    
    # Population
    
    
    particles = [
    [random.uniform(position_min, position_max) for j in range(dimension)]
    for i in range(population)
]
    velocity = [[0.0 for j in range(dimension)] for i in range(population)]

    # Particle's best position
    pbest_position = particles
    # Fitness
    pbest_fitness = [fitness_function(*p) for p in particles]
    # Index of the best particle
    gbest_index = np.argmin(pbest_fitness)
    # Global best particle position
    gbest_position = pbest_position[gbest_index]
    # Velocity (starting from 0 speed)


    # Loop for the number of generation
    for t in range(generation):
        # Stop if the average fitness value reached a predefined success criterion
        if np.average(pbest_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):
                # Update the velocity of each particle
                velocity[n] = update_velocity(
                    particles[n], velocity[n], pbest_position[n], gbest_position, w, c1, c2
                )
                
                
                
                Y.append(grafica(pbest_position[n]))
                
                
                # Move the particles to new position
                particles[n] = update_position(particles[n], velocity[n], rango)


        # Calculate the fitness value
        pbest_fitness = [fitness_function(*p) for p in particles]
        # Find the index of the best particle
        gbest_index = np.argmin(pbest_fitness)
        # Update the position of the best particle
        gbest_position = pbest_position[gbest_index]

    # Print the results
    # Al final de la función, llena el diccionario con los resultados
    resultados["Global Best Position"] = gbest_position
    resultados["Best Fitness Value"] = min(filter(lambda x: x >= 0, pbest_fitness))
    resultados["Average Particle Best Fitness Value"] = np.average(pbest_fitness)
    resultados["Number of Generation"] = t

    # Devuelve el diccionario de resultados
    return resultados


population = 100
dimension = 10
position_min = rango[0]
position_max = rango[1]
generation = 5000
fitness_criterion = 10e-2

w = [0.5]
c1 = [2]
c2 = [1]

combinaciiones = list(itertools.product(w,c1,c2))


# Función para crear y escribir en archivos de texto
def crear_y_escribir_archivo(nombre_archivo, contenido):
    with open(nombre_archivo, "w") as archivo:
        archivo.write(contenido)
        
        
for i in (combinaciiones):
    
    W = i[0]
    C1 = i[1]
    C2 = i[2]
    
    # ... (código anterior) ...

    # Crear una cadena para almacenar todos los resultados de una combinación
    resultados_combinacion = ""

    for j in range(0, len(primos)):
        random.seed(primos[j])

        resultados = pso_2d(
            population,
            dimension,
            position_min,
            position_max,
            generation,
            fitness_criterion,
            C1,
            C2,
            W
        )

        # Convierte los resultados en una cadena legible (ajusta esto según tus resultados)
        resultados_str = "\n".join([f"{clave}: {valor}" for clave, valor in resultados.items()])

        # Agrega los resultados de esta semilla a la cadena de resultados de la combinación
        resultados_combinacion += f"Resultados para Primo {primos[j]}:\n{resultados_str}\n\n"

    # Crea un nombre de archivo único basado en la combinación
    nombre_archivo = f"combinacion_{W}_{C1}_{C2}.txt"

    # Crea y escribe en el archivo TXT
    crear_y_escribir_archivo(nombre_archivo, resultados_combinacion)


# print(X)
# print(Y)

for i in range(2100):
    X.append(i)


datos = list(zip(X, Y))

# Ordenar la lista de tuplas en función de los valores de Y
datos_ordenados = sorted(datos, key=lambda x: x[1])

X_invertidos = list(reversed(X))

# Desempaquetar los valores ordenados de vuelta en X e Y
X_ordenados, Y_ordenados = zip(*datos_ordenados)

print(len(Y_ordenados))

plt.plot(X_invertidos, Y_ordenados)
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')
plt.title('RASTRIGIN')
plt.grid(True)
plt.show()
