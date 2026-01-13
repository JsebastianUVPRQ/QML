import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from src.models.benchmark import run_benchmark

def plot_decision_boundaries():
    # 1. Obtenemos los modelos entrenados del benchmark
    svc_c, svc_q, data = run_benchmark()
    
    X = data.X_test
    y = data.y_test

    # 2. Crear una malla (grid) de puntos para colorear el fondo
    h = 0.2  # Tamaño del paso en la malla (sube esto si tarda mucho, bájalo para más resolución)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 3. Predecir en toda la malla
    print("\n🎨 Generando contornos (esto tardará por la simulación cuántica)...")
    
    # Clásico
    Z_c = svc_c.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_c = Z_c.reshape(xx.shape)
    
    # Cuántico
    Z_q = svc_q.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_q = Z_q.reshape(xx.shape)

    # 4. Graficar
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Clásico
    ax[0].contourf(xx, yy, Z_c, cmap=plt.cm.coolwarm, alpha=0.8)
    ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax[0].set_title("Frontera de Decisión CLÁSICA (RBF)")
    
    # Plot Cuántico
    ax[1].contourf(xx, yy, Z_q, cmap=plt.cm.coolwarm, alpha=0.8)
    ax[1].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax[1].set_title("Frontera de Decisión CUÁNTICA (Ising)")

    plt.show()

if __name__ == "__main__":
    plot_decision_boundaries()