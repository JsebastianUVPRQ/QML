"""
Training Script.

Orquesta el flujo: Datos -> Feature Map -> Quantum Kernel -> SVM Clásico.
"""

import time
import sys
import os

# Ajuste de path para que encuentre nuestros módulos
sys.path.append(os.getcwd())

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.data.processor import CryptoFraudData
from src.quantum.feature_map import QuantumIsingMap

def main():
    print("🚀 Iniciando Pipeline de Detección de Anomalías Cuántico-Clásico")
    
    # 1. Preparar Datos
    # Usamos 2 qubits (dimensiones) para una simulación rápida en tu laptop
    data = CryptoFraudData(n_dim=2, sample_size=150) 
    data.load_and_process()
    
    # 2. Preparar el "Cerebro" Cuántico (Kernel)
    print("\n⚛️ Inicializando Kernel Cuántico (Ising Model)...")
    q_map = QuantumIsingMap(n_qubits=2, reps=2)
    
    # FidelityQuantumKernel calcula el overlap |<phi(x)|phi(y)>|^2
    qkernel = FidelityQuantumKernel(feature_map=q_map.get_circuit())
    
    # 3. Entrenamiento Híbrido
    # Usamos SVM clásico, pero la matriz de similitud la calcula el simulador cuántico
    model = SVC(kernel=qkernel.evaluate)
    
    print("\n⏳ Entrenando SVM (esto puede tardar unos segundos por la simulación)...")
    start = time.time()
    model.fit(data.X_train, data.y_train)
    end = time.time()
    
    print(f"✅ Modelo entrenado en {end - start:.2f} segundos.")
    
    # 4. Evaluación
    print("\n📊 Evaluando rendimiento...")
    score = model.score(data.X_test, data.y_test)
    print(f"Accuracy del modelo híbrido: {score:.2%}")
    
    y_pred = model.predict(data.X_test)
    print("\nReporte detallado:")
    print(classification_report(data.y_test, y_pred))

if __name__ == "__main__":
    main()