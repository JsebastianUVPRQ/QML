import sys
import os
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.getcwd())
from src.data.processor import CryptoFraudData
from src.quantum.feature_map import QuantumIsingMap
from src.quantum.backend import get_gpu_kernel

def run_benchmark():
    print("⚔️ INICIANDO BENCHMARK: Clásico vs. Cuántico")
    data = CryptoFraudData(n_dim=2, sample_size=140) # Ajustado para velocidad
    data.load_and_process()
    results = []

    # CLÁSICO
    print("\n🤖 Entrenando SVM Clásico (Kernel RBF)...")
    start = time.time()
    svc_classic = SVC(kernel='rbf', gamma='scale', C=1.0)
    svc_classic.fit(data.X_train, data.y_train)
    results.append({"Model": "Classical", "Accuracy": svc_classic.score(data.X_test, data.y_test)})
    print(f"   -> Accuracy: {results[-1]['Accuracy']:.2%}")

    # CUÁNTICO
    print("\n⚛️ Entrenando SVM Cuántico (Ising ZZ)...")
    q_map_obj = QuantumIsingMap(n_qubits=2, reps=2)
    qkernel = get_gpu_kernel(q_map_obj.get_circuit()) # Usa el backend robusto
    
    start = time.time()
    svc_quantum = SVC(kernel=qkernel.evaluate, C=1.0)
    svc_quantum.fit(data.X_train, data.y_train)
    results.append({"Model": "Quantum", "Accuracy": svc_quantum.score(data.X_test, data.y_test)})
    print(f"   -> Accuracy: {results[-1]['Accuracy']:.2%}")

    # GUARDAR
    if not os.path.exists("references"): os.makedirs("references")
    pd.DataFrame(results).to_csv("references/table_1_results.csv", index=False)
    
    return svc_classic, svc_quantum, data

if __name__ == "__main__":
    run_benchmark()