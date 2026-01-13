"""
Benchmark Script.
Compara SVM Clásico (RBF) vs SVM Cuántico (Ising Kernel).
"""
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.getcwd())
from src.data.processor import CryptoFraudData
from src.quantum.feature_map import QuantumIsingMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from src.quantum.backend import get_gpu_kernel

def run_benchmark():
    print("⚔️ INICIANDO BENCHMARK: Clásico vs. Cuántico")
    
    # 1. Cargar Datos
    # Usamos sample_size=200 para que la simulación cuántica no tarde horas
    data = CryptoFraudData(n_dim=2, sample_size=200)
    data.load_and_process()
    
    results = []

    # --- MODELO 1: CLÁSICO (RBF) ---
    print("\n🤖 Entrenando SVM Clásico (Kernel RBF)...")
    start = time.time()
    svc_classic = SVC(kernel='rbf', gamma='scale', C=1.0)
    svc_classic.fit(data.X_train, data.y_train)
    classic_time = time.time() - start
    
    y_pred_c = svc_classic.predict(data.X_test)
    acc_c = accuracy_score(data.y_test, y_pred_c)
    f1_c = f1_score(data.y_test, y_pred_c)
    
    results.append({
        "Model": "Classical (RBF)", 
        "Accuracy": acc_c, 
        "F1-Score": f1_c, 
        "Time (s)": classic_time
    })
    print(f"   -> Accuracy: {acc_c:.2%} | Tiempo: {classic_time:.4f}s")

    # --- MODELO 2: CUÁNTICO (Ising Kernel) ---
    print("\n⚛️ Entrenando SVM Cuántico (Ising ZZ)...")
    q_map = QuantumIsingMap(n_qubits=2, reps=2).get_circuit()
    qkernel = get_gpu_kernel(q_map)

    start = time.time()
    svc_quantum = SVC(kernel=qkernel.evaluate, C=1.0)
    svc_quantum.fit(data.X_train, data.y_train)
    quantum_time = time.time() - start
    
    y_pred_q = svc_quantum.predict(data.X_test)
    acc_q = accuracy_score(data.y_test, y_pred_q)
    f1_q = f1_score(data.y_test, y_pred_q)
    
    results.append({
        "Model": "Quantum (Ising)", 
        "Accuracy": acc_q, 
        "F1-Score": f1_q, 
        "Time (s)": quantum_time
    })
    print(f"   -> Accuracy: {acc_q:.2%} | Tiempo: {quantum_time:.4f}s")

    # --- VISUALIZACIÓN DE RESULTADOS ---
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 5))
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_res, x='Model', y='Accuracy', palette=['gray', 'purple'])
    plt.ylim(0, 1.1)
    plt.title("Comparación de Precisión")
    
    # Gráfica de F1 Score (Más importante en fraude)
    plt.subplot(1, 2, 2)
    sns.barplot(data=df_res, x='Model', y='F1-Score', palette=['gray', 'purple'])
    plt.ylim(0, 1.1)
    plt.title("Comparación F1-Score (Fraude)")
    
    plt.tight_layout()
    plt.show()

    return svc_classic, svc_quantum, data

if __name__ == "__main__":
    run_benchmark()