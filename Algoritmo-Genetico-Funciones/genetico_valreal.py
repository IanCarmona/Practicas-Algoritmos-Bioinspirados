import random
import math
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from statistics import mean


limInf = -2.04
limSup = 2.04
def fun1(*args):  # Rosenbrock
    # Optimizacion
    # ranges [-2.04, 2.04]
    n = len(args)
    if n == 2:  # Si son dos dimensiones, asume que son X e Y
        X, Y = args
        sum_term = (1 - X) ** 2 + 100 * (Y - X**2) ** 2
    else:
        sum_term = sum((1 - args[i])**2 + 100 * (args[i + 1] - args[i]**2)**2 for i in range(n - 1))
    return sum_term

# limInf = -32.76
# limSup = 32.76
# def fun1(*args):  # Ackley
#     n = len(args)
#     A = 20
#     B = 0.2
#     C = 2 * np.pi  # Utilizar np.pi en lugar de math.pi para trabajar con NumPy
#     if n == 2:  # Si son dos dimensiones, asume que son X e Y
#         x, y = args
#         sum_term1 = -A * np.exp(-B * np.sqrt(0.5 * (x**2 + y**2)))
#         sum_term2 = -np.exp(0.5 * (np.cos(C * x) + np.cos(C * y)))
#         result = sum_term1 + sum_term2 + A + np.exp(1)
#     else:
#         sum_term1 = -A * np.exp(-B * np.sqrt((1/n) * sum(x**2 for x in args)))
#         sum_term2 = -np.exp((1/n) * sum(np.cos(C * x) for x in args))
        
#         result = sum_term1 + sum_term2 + A + np.exp(1)

#     return result

def plotFunction(limInf, limSup, n):
    print(limInf, limSup, n)
    x1 = np.linspace(limInf, limSup, n)
    x2 = np.linspace(limInf, limSup, n)
    X, Y = np.meshgrid(x1, x2)
    FX = fun1(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # ax.contour3D(X, Y, FX, 50, cmap='binary')
    ax.plot_surface(X, Y, FX, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def codificar(numero, limInf, limSup):
    # Normalizar el número dentro del rango limInf a limSup
    numero_normalizado = max(limInf, min(numero, limSup))
    return numero_normalizado

def decodificar(valor, limInf, limSup):
    # Devolver el valor dentro del rango limInf a limSup
    return max(limInf, min(valor, limSup))

def poblacionInicial(tam, numVar, limInf, limSup):
    poblacion = []
    fx_a = []
    
    for _ in range(tam):
        X = [random.uniform(limInf, limSup) for _ in range(numVar)]
        fx = calculaAptitud(X)
        fx_a.append(fx)
        poblacion.append(X)
        
    return poblacion, fx_a

def calculaAptitud(X):
    return round(fun1(*X), 2)

def seleccion_ruleta(aptitudes):
    total_aptitudes = sum(aptitudes)
    probabilidades = [aptitud / total_aptitudes for aptitud in aptitudes]
    # Girar la ruleta (seleccionar un padre)
    ruleta = random.uniform(0, 1)
    acumulador = 0
    indice_seleccionado = 0
    for i, probabilidad in enumerate(probabilidades):
        acumulador += probabilidad
        if ruleta <= acumulador:
            indice_seleccionado = i
            break
    
    return indice_seleccionado

def defineParejas(tamPop, aptitudes):
    padres_seleccionados = set()
    while len(padres_seleccionados) < tamPop // 2:  
        indice_padre = seleccion_ruleta(aptitudes)
        indice_otro_padre = seleccion_ruleta(aptitudes)
        
        if indice_padre != indice_otro_padre and (indice_padre, indice_otro_padre) not in padres_seleccionados and (indice_otro_padre, indice_padre) not in padres_seleccionados:
            padres_seleccionados.add((indice_padre, indice_otro_padre))
    
    return [indice for pareja in padres_seleccionados for indice in pareja]

def cruzar_dos_puntos(padre1, padre2):
    punto_cruce = random.randint(0, len(padre1) - 1)
    hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
    hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
    
    return hijo1, hijo2

def creaHijos(pCruza, indices_padres, padres):
    hijos1 = []
    hijos2 = []
    
    for i in range(0, len(indices_padres), 2):
        padre1 = padres[indices_padres[i]]
        padre2 = padres[indices_padres[i + 1]]

        if random.uniform(0, 1) <= pCruza:
            hijo1, hijo2 = cruzar_dos_puntos(padre1, padre2)
            hijos1.append(hijo1)
            hijos2.append(hijo2)
                
    return hijos1, hijos2

def mutacion(cromosoma, porcMuta, limInf, limSup):
    nuevo_crom = []
    for gen in cromosoma:
        if random.uniform(0, 1) <= porcMuta:
            # Aplicar una pequeña perturbación al valor del gen
            perturbacion = random.uniform(-0.1, 0.1)  # Puedes ajustar el rango según sea necesario
            nuevo_gen = decodificar(gen + perturbacion, limInf, limSup)
        else:
            nuevo_gen = gen
        nuevo_crom.append(nuevo_gen)
    return nuevo_crom

def algoritmoGenetico(nVariables, limInf, limSup, tamPoblacion, porcCruza, porcMuta, gen_max):
    # genera poblacion inicial
    padres, fx = poblacionInicial(tamPoblacion, nVariables, limInf, limSup)
    #print("Poblacion inicial ", padres)
    print("===========================")
    mejores = []
    peores = []
    promedio = []
    mejorr = 0
    peorr = 10
    i = 0
    generaciones = 0
    # repite mientras no se alcance criterio de paro
    while((peorr - mejorr) >= 0.001 and generaciones <= gen_max):
        # crea parejas de padres
        generaciones = generaciones + 1
        i = i + 1
        indices = defineParejas(tamPoblacion, fx)
        #print("Parejas: ", indices)
        # aplica operador de cruza por cada pareja (para generar hijos)
        hijos1, hijos2 = creaHijos(porcCruza, indices, padres)
        hijos_mutados = []
        for cromosoma in hijos1 + hijos2:
            hijos_mutados.append(mutacion(cromosoma, porcMuta, limInf, limSup))
    
        # unir padres y descendientes
        nuevaPoblacion = padres + hijos_mutados
        #print("longitud de padres e hijos", len(nuevaPoblacion))
        valores_decodificados = nuevaPoblacion

        # print(valores_decodificados)
        aptitudes = [calculaAptitud(vector) for vector in valores_decodificados]
        # print(aptitudes)
        decodificados_aptitudes = list(zip(valores_decodificados, aptitudes))
        # Ordenar por las aptitudes
        sobrevivientes = sorted(decodificados_aptitudes, key=lambda x: x[1], reverse=False)
        #print("ordenada", sobrevivientes)
        #print("\n")
        padres = [p[0] for p in sobrevivientes[:tamPoblacion]]
        fx = [p[1] for p in sobrevivientes[:tamPoblacion]]
        #print(f"nuevos padres (sobrevivientes) {i}: {padres}")
        #print("\n")
        # registra los valores del mejor y peor individuo por generación
        mejores.append(fx[0])
        peores.append(fx[-1])
        # calcula la aptitud promedio de la población en cada generación
        prom = sum(fx)
        promedio.append(round((prom / len(padres)), 2))

    print("mejor solucion", padres[0])
    return mejores, peores, promedio, generaciones

def grafica(mejores, peores, promedio, generaciones, semilla):
    x = list(range(1, generaciones + 1))
    plt.figure(1)
    
    # Convertir listas a arreglos de NumPy
    mejores_array = np.array(mejores)
    peores_array = np.array(peores)
    
    # Concatenar los arreglos 'mejores_array' y 'peores_array'
    concatenados = np.concatenate((mejores_array, peores_array))
    
    plt.scatter(x, mejores, color="green", label="mejor")
    plt.plot(x, mejores, color="green")
    plt.scatter(x, peores, color="red", label="peor")
    plt.plot(x, peores, color="red")
    plt.scatter(x, promedio, color="blue", label="promedio")
    plt.plot(x, promedio, color="blue")
    plt.legend()
    plt.xlabel("Generaciones")
    plt.ylabel("Aptitud")
    
    # Calcular la desviación estándar de los arreglos concatenados
    desviacion_estandar = round(np.std(concatenados), 2)
    
    plt.title(f'Semilla {semilla} \n Grafica de convergencia con {generaciones} generaciones \n max:{peores[0]} min:{mejores[-1]} desviacion estandar: {desviacion_estandar}')
    
    plt.savefig(f'real_s{semilla}_rosembrok.jpg')
    plt.show()

# ...

# parametros de entrada
nVariables = 10
# el tamaño de la población debe ser un numero par
tamPoblacion = 100
porcCruza = 0.9
porcMuta = 0.1
gen_max = 150


# Lista de los primeros 20 números primos
primeros_primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
plotFunction(limInf, limSup, 50)
plt.show()

# Bucle para repetir el algoritmo 20 veces con diferentes semillas
for semilla in primeros_primos:
    print(f"\n*** Iteración con semilla {semilla} ***")
    random.seed(semilla)
    mejores, peores, promedio, generaciones= algoritmoGenetico(nVariables, limInf, limSup, tamPoblacion, porcCruza, porcMuta, gen_max)
    print(f'Mejores: {mejores}')
    print(f'Peores: {peores}')
    print(f'Promedio: {promedio}')
    grafica(mejores, peores, promedio, generaciones, semilla)
    

