# funciones de prueba

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




def codificar(numero, num_bits, limInf, limSup):
    # Normalizar el número dentro del rango 0 a 10
    numero_normalizado = max(limInf, min(numero, limSup))
    
    # Convertir a una escala entre 0 y 2^num_bits - 1
    valor_escala = int(numero_normalizado * (2**num_bits - 1) / 10)
    
    # Convertir a binario con la longitud deseada
    representacion_binaria = format(valor_escala, '0{:d}b'.format(num_bits))
    return representacion_binaria

def decodificar(binario, num_bits):
    # Convertir de binario a valor de escala entre 0 y 2^num_bits - 1
    valor_escala = int(binario, 2)
    
    # Escalar el valor al rango original (0 a 10)
    numero = valor_escala * 10 / (2**num_bits - 1)
    return round(numero, 2)

def poblacionInicial(tam, numVar, limInf, limSup):
    poblacion = []
    fx_a = []
    for _ in range(0, tam):
        X = []
        X_bin = []
        num_bits = int(numDigitos_cb(limInf, limSup))

        for _ in range(numVar):
            valor_x = round(random.uniform(limInf, limSup), 2)
            X.append(valor_x)
            
            valor_x_bin = codificar(valor_x, num_bits, limInf, limSup)
            X_bin.append(valor_x_bin)
            
        fx = calculaAptitud(X)
        fx_a.append(fx)
        poblacion.append(X_bin)
    return poblacion, fx_a

def calculaAptitud(X):
    return round(fun1(*X), 2)

def numDigitos_cb(limInf, limSup):
    return np.log2((limSup*(10**2)-limInf*(10**2)))

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
    numdig = len(padre1)
    punto_cruce1 = random.randint(0, numdig - 1)
    punto_cruce2 = random.randint(0, numdig - 1)
    
    while punto_cruce1 == punto_cruce2:
        punto_cruce2 = random.randint(0, numdig - 1)
    
    punto_cruce1, punto_cruce2 = min(punto_cruce1, punto_cruce2), max(punto_cruce1, punto_cruce2)
    
    hijo1 = padre1[:punto_cruce1] + padre2[punto_cruce1:punto_cruce2] + padre1[punto_cruce2:]
    hijo2 = padre2[:punto_cruce1] + padre1[punto_cruce1:punto_cruce2] + padre2[punto_cruce2:]
    
    return hijo1, hijo2

def creaHijos(pCruza, indices_padres, padres):
    hijos1 = []
    hijos2 = []
    
    dim = len(padres[0])
    numdig = len(padres[0][0])

    for i in range(0, len(indices_padres), 2):
        padre1 = padres[indices_padres[i]]
        padre2 = padres[indices_padres[i + 1]]

        if random.uniform(0, 1) <= pCruza:
            for x in range(dim):
                p1 = padre1[x]
                p2 = padre2[x]

                h1, h2 = cruzar_dos_puntos(p1, p2)

                hijos1.append(h1)
                hijos2.append(h2)
    return hijos1, hijos2

def mutacion(cromosoma, porcMuta):
    nuevo_crom = []
    for gen in cromosoma:
        if random.uniform(0, 1) <= porcMuta:
            # Invertir los valores '0' por '1' y '1' por '0'
            nuevo_gen = ''.join(['1' if bit == '0' else '0' for bit in gen])
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
        hijos_redimencionados1 = np.reshape(hijos1, (len(hijos1)//nVariables, nVariables))
        hijos_redimencionados2 = np.reshape(hijos2, (len(hijos2)//nVariables, nVariables))
        hijos_completos = np.concatenate((hijos_redimencionados1, hijos_redimencionados2), axis = 0)
        # print("longitud de hijos", len(hijos))
        # print(hijos)

        # aplica operador de mutacion
        hijos_mutados = []
        for cromosoma in hijos_completos:
            hijos_mutados.append(mutacion(cromosoma, porcMuta))
    

        # unir padres y descendientes
        nuevaPoblacion = np.concatenate((padres, hijos_mutados), axis = 0)
        #print("longitud de padres e hijos", len(nuevaPoblacion))
        # print("nueva poblacion", nuevaPoblacion)
        
        valores_decodificados = []

        for cromosoma in nuevaPoblacion:
            valores_decodificados.append([decodificar(gen, len(gen)) for gen in cromosoma])

        # print(valores_decodificados)
        
        aptitudes = [calculaAptitud(vector) for vector in valores_decodificados]
        
        # print(aptitudes)
        
        decodificados_aptitudes = list(zip(valores_decodificados, aptitudes))

        # Ordenar por las aptitudes
        sobrevivientes = sorted(decodificados_aptitudes, key=lambda x: x[1], reverse=False)
        
        #print("ordenada", sobrevivientes)
        #print("\n")
        padres = sobrevivientes[0:tamPoblacion]
        #print(f"nuevos padres (sobrevivientes) {i}: {padres}")
        #print("\n")
        
        # registra los valores del mejor y peor individuo por generación
        mejores.append(padres[0][1])
        peores.append(padres[-1][1])
        #print(padres[0][1])
        mejorr = padres[0][1]
        peorr = padres[-1][1]    
        
        
        # calcula la aptitud promedio de la población en cada generación
        prom = 0
        for p in padres:
            prom += p[1]
        promedio.append(round((prom / len(padres)), 2))

        padre = []
        fxx = []
        
        for elemento in padres:
            padre.append(elemento[0])
            fxx.append(elemento[1])

        padres = padre
        
        
        
        
        num_bits = int(numDigitos_cb(limInf, limSup))
        vectores_codificados = []
        for vector in padres:
            vector_codificado = [codificar(valor, num_bits, limInf, limSup) for valor in vector]
            vectores_codificados.append(vector_codificado)
        
        
        
        padres = vectores_codificados
        
        # print(padres)
        fx = fxx
        
    

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
    
    plt.savefig(f'binario_s{semilla}_rosembrok.jpg')
    
    plt.show()
    

    

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
    


