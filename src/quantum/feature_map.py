"""
Quantum Feature Maps Module.

Este módulo define los circuitos variacionales que mapean datos clásicos
al espacio de Hilbert (Quantum Kernel).
Basado en la física del Modelo de Ising.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
import numpy as np

class QuantumIsingMap:
    """
    Generador de mapas de características basado en interacciones ZZ (Modelo de Ising).
    
    El circuito implementa la transformación unitaria:
    U(x) = exp(-i * (sum_j phi(x_j)Zj + sum_jk phi(x_j, x_k)ZjZk))
    """
    
    def __init__(self, n_qubits: int, reps: int = 2):
        """
        Inicializa el mapa cuántico.

        Args:
            n_qubits (int): Dimensión del espacio de características (y número de qubits).
                            Si usamos PCA para reducir a 2 variables, n_qubits=2.
            reps (int): Profundidad del circuito. Cuántas veces repetimos la evolución temporal.
                        Mayor reps = Mayor entrelazamiento (pero más ruido en hardware real).
        """
        self.n_qubits = n_qubits
        self.reps = reps
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """
        Construye el objeto QuantumCircuit usando la librería de Qiskit.
        """
        # Usamos 'linear' entanglement: el qubit 0 habla con el 1, el 1 con el 2...
        # Esto simula una cadena de espines 1D (común en materia condensada).
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, 
            reps=self.reps, 
            entanglement='linear'
        )
        return feature_map

    def get_circuit(self) -> QuantumCircuit:
        """Retorna el circuito cuántico construido."""
        return self._circuit

    def draw(self):
        """Imprime una representación visual del circuito."""
        print(self._circuit.decompose().draw(output='text'))

# Bloque de prueba (solo se ejecuta si corres este archivo directamente)
if __name__ == "__main__":
    print("🧪 Test unitario del Feature Map (Ising Model)...")
    q_map = QuantumIsingMap(n_qubits=2, reps=1)
    q_map.draw()
    print("\n✅ Circuito generado correctamente basedo en Hamiltonianos ZZ.")