
import matplotlib.pyplot as plt
import random
import numpy as np

def redimensionar_lista_a_matriz(lista, n):
    if len(lista) != n*n:
        return "El número de elementos en la lista no coincide con una matriz de nxn."
    
    matriz = [lista[i:i+n] for i in range(0, len(lista), n)]
    return matriz

def calcular_suma_magica(n):
    return (n * (n**2 + 1)) // 2

def crear_cuadrado_magico(n):
    numeros = list(range(1, n * n + 1))
    random.shuffle(numeros)
    cuadrado_magico = [0] * (n * n)
    
    i, j = 0, n // 2
    for num in numeros:
        cuadrado_magico[i * n + j] = num
        i, j = (i - 1) % n, (j + 1) % n
        if cuadrado_magico[i * n + j]:
            i = (i + 1) % n
    
    return cuadrado_magico
    

def calculaAptitudMin(X, n):
    
    arr_2d = redimensionar_lista_a_matriz(X, n)
    
    
    error = []
    
    for i in range(n):
        suma = 0
        for j in range(n):
            suma += arr_2d[i][j]
            
        error.append(abs(calcular_suma_magica(n) - suma))
    

    for i in range(n):
        suma = 0
        for j in range(n):
            suma += arr_2d[j][i]
        error.append(abs(calcular_suma_magica(n) - suma))
    
            
    suma = 0
    for i in range(n):
        suma += arr_2d[i][i]
    error.append(abs(calcular_suma_magica(n) - suma))
    
    
            
    suma = 0
    for i in range(n):
        suma += arr_2d[i][n - 1 - i]
    error.append(abs(calcular_suma_magica(n) - suma))
    
    
    
    return sum(error)





def calculaAptitudMax(X, n):
    arr_2d = redimensionar_lista_a_matriz(X, n)
    
    contador = 0
    
    for i in range(n):
        suma = 0
        for j in range(n):
            suma += arr_2d[i][j]
        
        if (suma == calcular_suma_magica(n)):
            contador = contador + 1

    for i in range(n):
        suma = 0
        for j in range(n):
            suma += arr_2d[j][i]
        
        if (suma == calcular_suma_magica(n)):
            contador = contador + 1
            
    suma = 0
    for i in range(n):
        suma += arr_2d[i][i]
    
    if (suma == calcular_suma_magica(n)):
            contador = contador + 1
            
    suma = 0
    for i in range(n):
        suma += arr_2d[i][n - 1 - i]
    
    if (suma == calcular_suma_magica(n)):
            contador = contador + 1
    
    return contador

def poblacionInicial(n, tamPoblacion):
    poblacion = []
    
    for _ in range(tamPoblacion):
        cuadrado = crear_cuadrado_magico(n)
        aptitud = calculaAptitudMin(cuadrado, n)
        poblacion.append([cuadrado, aptitud])
    
    return poblacion


def seleccion_ruleta(aptitudes):
    total_aptitudes = sum(aptitudes)
    probabilidades = [aptitud / total_aptitudes if total_aptitudes != 0 else 1 / len(aptitudes) for aptitud in aptitudes]
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


def calcular_aptitudes(poblacion):
    aptitudes = []
    for subarreglo in poblacion:
        ultimo = subarreglo[-1]
        aptitudes.append(ultimo)
    return aptitudes



# crea poblacion de descendientes
def creaHijos(pCruza, indices, padres, tamPob):
    hijos = []
    j = 0
    while j < tamPob:
        if random.uniform(0, 1) <= pCruza:
            p1 = padres[indices[j]][0]
            p2 = padres[indices[j + 1]][0]
            # print("padre1",p1)
            # print("padre2",p2)
            h1, h2 = cruza_pmx(p1, p2)
            # print("hijo1", h1)
            # print("hijo2", h2)
            hijos.append(h1)
            hijos.append(h2)
        j += 2
    return hijos

def cruza_pmx(padre1, padre2):
    tamano = len(padre1)
    hijo1 = [-1] * tamano
    hijo2 = [-1] * tamano

    punto1 = random.randint(0, tamano - 1)
    punto2 = random.randint(0, tamano - 1)

    while punto1 == punto2:
        punto2 = random.randint(0, tamano - 1)

    if punto1 > punto2:
        punto1, punto2 = punto2, punto1

    hijo1[punto1:punto2+1] = padre1[punto1:punto2+1]
    hijo2[punto1:punto2+1] = padre2[punto1:punto2+1]

    for i in range(punto1, punto2+1):
        if padre2[i] not in hijo1:
            indice = padre2.index(padre1[i])
            while hijo1[indice] != -1:
                indice = padre2.index(padre1[indice])
            hijo1[indice] = padre2[i]

        if padre1[i] not in hijo2:
            indice = padre1.index(padre2[i])
            while hijo2[indice] != -1:
                indice = padre1.index(padre2[indice])
            hijo2[indice] = padre1[i]

    for i in range(tamano):
        if hijo1[i] == -1:
            hijo1[i] = padre2[i]
        if hijo2[i] == -1:
            hijo2[i] = padre1[i]

    return hijo1, hijo2

def mutacion_intercambio_reciproco(lista, porcMuta, aptitud, n, lista_tabu,tam_tabu):
    if random.uniform(0, 1) <= porcMuta:
        
        if aptitud >= calcular_suma_magica(n) // 2:
            
            while True: 
                punto1 = random.randint(0, len(lista) - 1)
                punto2 = random.randint(0, len(lista) - 1)

                while punto1 == punto2:
                    punto2 = random.randint(0, len(lista) - 1)

                # Verificar si el intercambio está en la lista tabú
                if (punto1, punto2) not in lista_tabu and (punto2, punto1) not in lista_tabu:
                    lista[punto1], lista[punto2] = lista[punto2], lista[punto1]
                    # Agregar el intercambio a la lista tabú
                    lista_tabu.append((punto1, punto2))
                    if len(lista_tabu) > tam_tabu: 
                        lista_tabu.pop(0)
                    break
                
                if random.uniform(0, 1) <= 0.3: 
                    return lista
        
        else:
            while True: 
                punto1 = random.randint(0, len(lista) - 1)
                punto2 = random.randint(0, len(lista) - 1)

                while punto1 == punto2:
                    punto2 = random.randint(0, len(lista) - 1)

                # Verificar si el intercambio está en la lista tabú
                if (punto1, punto2) not in lista_tabu and (punto2, punto1) not in lista_tabu:
                    lista[punto1], lista[punto2] = lista[punto2], lista[punto1]
                    # Agregar el intercambio a la lista tabú
                    lista_tabu.append((punto1, punto2))
                    if len(lista_tabu) > tam_tabu:  
                        lista_tabu.pop(0)
                    break
                
                if random.uniform(0, 1) <= 0.7: 
                    return lista
    return lista
def algoritmoGenetico(n, tamPoblacion, porcCruza, porcMuta, generaciones, tam_tabu, max_generaciones_sin_mejora):
    lista_tabu = []
    padres = poblacionInicial(n, tamPoblacion)
    print("Poblacion inicial ", padres)
    print("===========================")
    mejores = []
    peores = []
    promedio = []
    contador = 0
    generaciones_sin_mejora = 0  # Contador para llevar el seguimiento de las generaciones sin mejora
    mejor_solucion = None

    while True:
        contador = contador + 1
        indices = defineParejas(tamPoblacion, calcular_aptitudes(padres))
        hijos = creaHijos(porcCruza, indices, padres, tamPoblacion)
        hijos2 = []

        for hijo in hijos:
            mutado = mutacion_intercambio_reciproco(hijo, porcMuta, calculaAptitudMin(hijo, n), n, lista_tabu, tam_tabu)
            hijos2.append([mutado, calculaAptitudMin(mutado, n)])

        nuevaPoblacion = padres + hijos2
        sobrevivientes = sorted(nuevaPoblacion, key=lambda x: x[1], reverse=False)
        padres = sobrevivientes[0:tamPoblacion]

        # Registro de valores del mejor y peor individuo por generación
        mejores.append(padres[0][1])
        peores.append(padres[-1][1])

        # Cálculo de la aptitud promedio de la población en cada generación
        prom = 0
        for p in padres:
            prom += p[1]
        promedio.append(prom / len(padres))

        #print(f"Generación {contador}: Mejor aptitud = {padres[0][1]}, Peor aptitud = {padres[-1][1]}")

        # Verificar si la aptitud ha mejorado
        if padres[0][1] == 0:
            print(f"La mejor solución fue encontrada en la generación {contador}")
            mejor_solucion = padres[0]
            break

        # Verificar si la aptitud no ha mejorado en 'max_generaciones_sin_mejora' generaciones
        if contador > max_generaciones_sin_mejora and mejores[-1] == mejores[-max_generaciones_sin_mejora]:
            generaciones_sin_mejora += 1
        else:
            generaciones_sin_mejora = 0

        # Si no ha mejorado en 'max_generaciones_sin_mejora' generaciones, quedarse con el mejor y hacer una nueva población
        if generaciones_sin_mejora == max_generaciones_sin_mejora:
            print(f"Generaciones sin mejora durante {max_generaciones_sin_mejora} generaciones.")
            mejor_solucion = padres[0]
            padres = [padres[-1]] + poblacionInicial(n, tamPoblacion - 1)
            generaciones_sin_mejora = 0
        if contador >= generaciones:
            break

    print("Mejor solución encontrada:")
    print(mejor_solucion[0])  # La mejor solución está en la posición 0 de la tupla en mejor_solucion
    print("Aptitud de la mejor solución:")
    print(mejor_solucion[1])

    return mejores, peores, promedio, contador



def grafica(mejores, peores, promedio, generaciones, n, semilla):
    x = list(range(1, generaciones + 1))
    
    plt.figure(1)
    
    # Convertir listas a arreglos de NumPy
    mejores_array = np.array(mejores)
    peores_array = np.array(peores)
    
    # Concatenar los arreglos 'mejores_array' y 'peores_array'
    concatenados = np.concatenate((mejores_array, peores_array))    
    
    plt.scatter(x, mejores, color="green", label="mejor", s = 10)
    plt.plot(x, mejores, color="green")
    plt.scatter(x, peores, color="red", label="peor", s = 1)
    # plt.plot(x, peores, color="red")
    plt.scatter(x, promedio, color="blue", label="promedio", s = 1)
    # plt.plot(x, promedio, color="blue")
    plt.legend()
    plt.xlabel("Generaciones")
    plt.ylabel("Aptitud")
    
    # Calcular la desviación estándar de los arreglos concatenados
    desviacion_estandar = round(np.std(concatenados), 2)
    
    plt.title(f'Semilla {semilla} \n Grafica de convergencia con {generaciones} generaciones \n max:{peores[0]} min:{mejores[-1]} desviacion estandar: {desviacion_estandar}')
    
    plt.savefig(f'{semilla}_funcion_min_n={n}.jpg')
    plt.show()


n = 10
tamPoblacion = 100
porcCruza = 0.9
porcMuta = 0.4
generaciones = 30000
tam_tabu = 2
max_generaciones_sin_mejora = 300



# #PRUEBAS NORMALES
# lista_tabu = []
# mejores, peores, promedio, generaciones2 = algoritmoGenetico(n, tamPoblacion, porcCruza, porcMuta, generaciones,tam_tabu, max_generaciones_sin_mejora)
# numero_magico = calcular_suma_magica(n)
# print(f"Número mágico: {numero_magico}")
# #print(f'Mejores aptitudes: {mejores}')
# #print(f'Peores aptitudes: {peores}')
# # print(f'Promedio de aptitudes: {promedio}')

    
# grafica(mejores, peores, promedio, generaciones2, n)

# PRUEBA FINAL
primeros_primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Bucle para repetir el algoritmo 20 veces con diferentes semillas
for semilla in primeros_primos:
    print(f"\n*** Iteración con semilla {semilla} ***")
    random.seed(semilla)
    lista_tabu = []
    mejores, peores, promedio, generaciones2 = algoritmoGenetico(n, tamPoblacion, porcCruza, porcMuta, generaciones,tam_tabu, max_generaciones_sin_mejora)

    # print(f'Mejores aptitudes: {mejores}')
    # print(f'Peores aptitudes: {peores}')
    # print(f'Promedio de aptitudes: {promedio}')
    
    grafica(mejores, peores, promedio, generaciones2, n, semilla)
    

#si se queda estancado, después de un número de generaciones, quedarse con el mejor y hacer una nueva población