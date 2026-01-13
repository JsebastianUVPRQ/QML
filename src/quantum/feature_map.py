from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

class QuantumIsingMap:
    def __init__(self, n_qubits: int, reps: int = 2):
        self.n_qubits = n_qubits
        self.reps = reps
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits, 
            reps=self.reps, 
            entanglement='linear'
        )
        return feature_map

    def get_circuit(self) -> QuantumCircuit:
        return self._circuit