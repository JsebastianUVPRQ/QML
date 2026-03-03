

### Instalación del Stack "Quantum-Classic" 🛠️

Abre tu terminal. Te recomiendo crear un entorno virtual para no ensuciar tu instalación principal de Python.

```bash
# 1. Crear entorno virtual (opcional pero recomendado)
python -m venv qml-env
source qml-env/bin/activate  # En Windows: qml-env\Scripts\activate

# 2. Instalar el núcleo científico clásico
pip install numpy pandas scikit-learn matplotlib seaborn

# 3. Instalar el stack de IBM Qiskit
# qiskit: el SDK base
# qiskit-machine-learning: conectores para kernels cuánticos
# qiskit-aer: simulador de alto rendimiento (necesario para correrlo en tu CPU)
pip install qiskit qiskit-machine-learning qiskit-aer

```

---

### main script

Este script hace el trabajo sucio: carga los datos, balancea las clases (porque hay muy pocos fraudes vs. transacciones reales) y aplica PCA para reducir la dimensionalidad de 29 variables a solo 2 o 4 (que son las que "caben" en una simulación cuántica casera).

Crea un archivo llamado `01_data_prep.py` y pega esto:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Configuración
# Reducimos a 2 dimensiones para poder visualizarlo y simularlo rápido en tu laptop
N_DIM = 2 
SAMPLE_SIZE = 200  # Empezamos con pocos datos para probar el circuito cuántico

print("🔄 Cargando o generando dataset...")

# INTENTO DE CARGA: Si tienes el archivo de Kaggle 'creditcard.csv' en la misma carpeta
try:
    df = pd.read_csv('creditcard.csv')
    print("✅ Dataset real encontrado.")
    
    # Tomamos una muestra balanceada (50% fraude, 50% legal) para el entrenamiento
    fraud = df[df.Class == 1]
    legal = df[df.Class == 0].sample(n=len(fraud), random_state=42)
    
    # Concatenamos y mezclamos
    data = pd.concat([fraud, legal]).sample(frac=1, random_state=42)
    
    # Separamos Features (X) y Target (y)
    # Las columnas V1...V28 ya son PCA, pero 'Time' y 'Amount' no. 
    # Para este ejercicio, usaremos todo y re-aplicaremos PCA para bajar a N_DIM.
    X_raw = data.drop(['Class'], axis=1).values
    y = data['Class'].values

except FileNotFoundError:
    print("⚠️ No encontré 'creditcard.csv'. Generando datos sintéticos para prueba...")
    from sklearn.datasets import make_classification
    X_raw, y = make_classification(n_samples=SAMPLE_SIZE, n_features=30, 
                                   n_informative=2, n_redundant=0, 
                                   n_clusters_per_class=1, weights=[0.5, 0.5], 
                                   random_state=42)

# --- PREPROCESAMIENTO ---

# 1. Escalar datos (StandardScaler es vital para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2. Reducción de Dimensionalidad (PCA) -> Autovectores
# Bajamos de 30 dimensiones a N_DIM (ej. 2 qubits)
pca = PCA(n_components=N_DIM)
X_pca = pca.fit_transform(X_scaled)

# 3. Escalar de nuevo entre -1 y 1 (o 0 y 2pi) para las rotaciones cuánticas
# Los qubits rotan con ángulos, así que normalizar a rangos conocidos ayuda.
minmax = MinMaxScaler(feature_range=(-1, 1))
X_final = minmax.fit_transform(X_pca)

# Dividir en Train y Test (usamos una muestra pequeña para Qiskit)
X_train, X_test, y_train, y_test = train_test_split(
    X_final[:SAMPLE_SIZE], y[:SAMPLE_SIZE], test_size=0.3, random_state=42
)

print(f"\n📊 Datos listos para el Circuito Cuántico:")
print(f"   Shape original: {X_raw.shape}")
print(f"   Shape reducido (PCA): {X_final.shape}")
print(f"   Muestras para entrenamiento: {X_train.shape[0]}")

# --- VISUALIZACIÓN CLÁSICA ---
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='blue', label='Legal', alpha=0.6)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='red', label='Fraude', alpha=0.6)
plt.title(f'Espacio Latente (PCA {N_DIM}D) - Antes del Kernel Cuántico')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

```

### ¿Qué acabas de hacer? 🧠

1. **Ingeniería de Características:** Convertiste un problema de 30 dimensiones (intratable para simuladores caseros) en uno de 2 dimensiones usando **PCA**.
2. **Normalización:** Preparaste los datos numéricos para que puedan ser interpretados como *ángulos de rotación* en la esfera de Bloch del qubit.

### Siguiente paso 👇

Ejecuta este código. Deberías ver una gráfica con puntos azules y rojos.

Si ves que los puntos rojos y azules están muy mezclados (difíciles de separar con una línea recta), **¡felicidades!** 🎉 Acabas de demostrar por qué necesitamos un **Kernel Cuántico**: para proyectar estos puntos a un espacio más complejo donde sí se puedan separar.