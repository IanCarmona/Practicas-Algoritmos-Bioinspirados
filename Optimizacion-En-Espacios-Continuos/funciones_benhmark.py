from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.exp(1) + 20

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rastrigin(x, y):
    return x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) + 20

def griewank(x, y):
    return (x**2 + y**2) / 4000 - (np.cos(x) * np.cos(y / np.sqrt(2))) + 1

def plotFunction(func, limInf, limSup, pts, nameFunction):
    X = np.linspace(limInf[0], limSup[0], pts)
    Y = np.linspace(limInf[1], limSup[1], pts)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=cm.hsv, edgecolor="darkred", linewidth=0.1
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")

    plt.savefig(nameFunction + ".jpg")
    plt.show()

# Definir los límites y puntos para las funciones
limInf = [-2, -2]
limSup = [2, 2]
pts = 100

# Graficar las funciones
plotFunction(ackley, limInf, limSup, pts, "ackley")
plotFunction(rosenbrock, limInf, limSup, pts, "rosenbrock")
plotFunction(rastrigin, limInf, limSup, pts, "rastrigin")
plotFunction(griewank, limInf, limSup, pts, "griewank")
