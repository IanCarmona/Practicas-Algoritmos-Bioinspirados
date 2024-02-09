# funciones de prueba
from numpy import *
from math import *
from pylab import *
import matplotlib.pyplot as plt


def f1(x):  # intervalo [-1.0, 1.0]
    # y = 65 - 0.75 / (1 + x**2) - (0.65 * x * math.atan(1/x))
    y = 65 - 0.75 / (1 + x**2) - (0.65 * x * np.arctan(1.0 / x))
    return y


def f2(x):  # intervalo [-10.0, 10.0]
    y = (1 - x) ** 4 - (2 * x + 10) ** 2
    return y


def f3(x):  # intervalo [-1.0, 1.0]
    y = 3 * x**2 + 12.0 / x**3 - 5.0
    return y


def f4(x):  # intervalo [-5.0, 5.0]
    y = x * (x - 1.5)
    return y


def f5(x):  # intervalo [-1.0, 1.0]
    y = 3 * x**4 + (x - 1) ** 2
    return y


def f6(x):  # intervalo [-5.0, 6.0]
    y = 10 * x**3 - 2 * x - 5 * exp(x)
    return y


def buscaMinimo(x1, x2, x3, f):
    if f(x1) >= f(x2) and f(x2) <= f(x3):
        return 1
    else:
        return 0


def exhaustiva(a, b, n, f):
    x1 = a
    deltaX = (b - a) / n
    x2 = x1 + deltaX
    x3 = x2 + deltaX
    iteraciones = 0
    while True:
        iteraciones = iteraciones + 1
        if buscaMinimo(x1, x2, x3, f) == 1:
            print("El mínimo se encuentra entre", x1, " y ", x3)
            break
        else:
            x1 = x2
            x2 = x3
            x3 = x2 + deltaX
        if x3 > b:
            break
        # else:
        # print("No existe un mínimo en (a,b) o un punto extremo (a ó b) es el mínimo")
    print("Numero de iteraciones", iteraciones)




# p1 = plot(x, f1(x))
# plt.grid()
# n = 10  # 1000 -> 904, 100-> 90, 10 -> 9



def divisionIntervalos(a, b, eps, fun):
    xm = (a + b) / 2
    L0 = L = b - a
    fun(eps)
    iteraciones = 0
    while True:
        iteraciones = iteraciones + 1
        x1 = a + L / 4
        x2 = b - L / 4
        if fun(x1) < fun(eps):
            b = xm
            xm = x1
        elif fun(x2) < fun(eps):
            a = xm
            xm = x2
        else:
            a = x1
            b = x2
        L = b - a
        if abs(L) < eps:
            break
    print("El óptimo se encuentra en el intervalo", a, b)
    print("numero de iteraciones", iteraciones)


prueba_busqueda_exhaustiva = [10, 100, 1000, 10000]
prueba_division_intervalos = [0.1, 0.01, 0.001, 0.0001]

print("BÚSQUEDA EXHAUSTIVA")
for i in prueba_busqueda_exhaustiva:
    print(f"PRUEBA DE LA FUNCIÓN CON VALOR: {i}\n")
    print(f"Resultado f1 {i}: {exhaustiva(-1, 1, i, f1)}")
    print(f"Resultado f2 {i}: {exhaustiva(-10, 10, i, f2)}")
    print(f"Resultado f3 {i}: {exhaustiva(-1, 1, i, f3)}")
    print(f"Resultado f4 {i}: {exhaustiva(-5, 5, i, f4)}")
    print(f"Resultado f5 {i}: {exhaustiva(-1, 1, i, f5)}")
    print(f"Resultado f6 {i}: {exhaustiva(-5, 6, i, f6)}\n")

print("DIVISIÓN INTERVALOS ")
for i in prueba_division_intervalos:
    print(f"PRUEBA DE DIVISIÓN EN EL INTERVALO: {i}\n")
    print(f"Resultado f1 {i}: {divisionIntervalos(-1, 1, i, f1)}")
    print(f"Resultado f2 {i}: {divisionIntervalos(-10, 10, i, f2)}")
    print(f"Resultado f3 {i}: {divisionIntervalos(-1, 1, i, f3)}")
    print(f"Resultado f4 {i}: {divisionIntervalos(-5, 5, i, f4)}")
    print(f"Resultado f5 {i}: {divisionIntervalos(-1, 1, i, f5)}")
    print(f"Resultado f6 {i}: {divisionIntervalos(-5, 6, i, f6)}\n")
