import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Configuración de Estilo Académico
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "stix", # Estilo similar a LaTeX real
    "figure.figsize": (10, 5),
    "lines.linewidth": 1.5
})

sys.path.append(os.getcwd())
from src.models.benchmark import run_benchmark

def plot_paper_figure():
    print("\n🎨 Generando Figura 1 para el Paper (Alta Calidad)...")
    
    # 1. Ejecutar Benchmark para obtener modelos entrenados
    # Nota: Esto imprimirá la tabla de resultados en la consola también.
    svc_c, svc_q, data = run_benchmark()
    
    X = data.X_test
    y = data.y_test

    # 2. Configurar Malla de Alta Resolución
    # h = 0.1 da mejor calidad que 0.2, pero tarda más. 
    # Ajusta según tu paciencia y potencia de GPU.
    h = 0.1  
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 3. Predicciones
    print("   -> Calculando superficie de decisión Clásica...")
    Z_c = svc_c.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_c = Z_c.reshape(xx.shape)
    
    print("   -> Calculando superficie de decisión Cuántica (GPU)...")
    Z_q = svc_q.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_q = Z_q.reshape(xx.shape)

    # 4. Graficar
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Paleta 'RdBu' es excelente para divergencia (Rojo-Azul) y 'Ising-like'
    cmap = plt.cm.RdBu_r 

    # --- Panel A: Clásico ---
    # Usamos contourf para las regiones y contour para la línea negra de decisión
    ax[0].contourf(xx, yy, Z_c, cmap=cmap, alpha=0.3)
    ax[0].contour(xx, yy, Z_c, colors='k', linewidths=0.5, alpha=0.5)
    
    # Puntos de datos (con bordes blancos para resaltar)
    # Clase 0 (Legal) = Azul, Clase 1 (Fraude) = Rojo
    ax[0].scatter(X[y==0, 0], X[y==0, 1], c='navy', s=30, edgecolors='w', label='Clase 0 (Normal)', alpha=0.7)
    ax[0].scatter(X[y==1, 0], X[y==1, 1], c='maroon', s=30, edgecolors='w', marker='^', label='Clase 1 (Anomalía)', alpha=0.9)
    
    ax[0].set_title(r"(a) Kernel RBF Clásico ($\gamma=scale$)", pad=15)
    ax[0].set_xlabel(r"Componente Principal 1 ($PC_1$)")
    ax[0].set_ylabel(r"Componente Principal 2 ($PC_2$)")
    ax[0].legend(loc='lower left', frameon=True, fancybox=False, edgecolor='k')

    # --- Panel B: Cuántico ---
    ax[1].contourf(xx, yy, Z_q, cmap=cmap, alpha=0.3)
    ax[1].contour(xx, yy, Z_q, colors='k', linewidths=0.5, alpha=0.5)
    
    ax[1].scatter(X[y==0, 0], X[y==0, 1], c='navy', s=30, edgecolors='w', alpha=0.7)
    ax[1].scatter(X[y==1, 0], X[y==1, 1], c='maroon', s=30, edgecolors='w', marker='^', alpha=0.9)
    
    ax[1].set_title(r"(b) Kernel Cuántico Ising ($n=2, d=2$)", pad=15)
    ax[1].set_xlabel(r"Componente Principal 1 ($PC_1$)")

    plt.tight_layout()
    
    # 5. Guardar en formato vectorial (PDF) para el paper y PNG para web
    save_path = "references/figure_1_decision_boundary"
    if not os.path.exists("references"): os.makedirs("references")
    
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=300)
    
    print(f"\n✅ Gráficos guardados en carpeta 'references/':")
    print(f"   - {save_path}.pdf (Insertar en LaTeX)")
    print(f"   - {save_path}.png (Para previsualizar)")
    plt.show()

if __name__ == "__main__":
    plot_paper_figure()