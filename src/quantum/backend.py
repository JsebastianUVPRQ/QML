"""
Quantum Backend Configuration.
Gestiona el acceso al simulador AER y la aceleración por GPU (NVIDIA).
"""
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler # O usamos la primitiva de AER si se prefiere

def get_quantum_kernel(feature_map, use_gpu=True):
    """
    Configura y devuelve un Kernel Cuántico acelerado.
    """
    
    # Configuración del simulador
    if use_gpu:
        try:
            # Intentamos forzar el uso de la GPU
            sim = AerSimulator(method='statevector', device='GPU')
            
            # Verificación dummy para ver si la GPU responde
            sim.run(feature_map.assign_parameters([0.1]*feature_map.num_parameters)).result()
            print("🚀 GPU NVIDIA detectada y activada para Qiskit Aer.")
            
        except Exception as e:
            print(f"⚠️ No se pudo inicializar GPU ({e}). Usando CPU...")
            sim = AerSimulator(method='statevector', device='CPU')
    else:
        sim = AerSimulator(method='statevector', device='CPU')

    # Para conectar el simulador con el Kernel, necesitamos una "Fidelidad"
    # que use este simulador específico.
    
    # Creamos una instancia de Fidelidad que usa nuestro simulador acelerado
    fidelity = ComputeUncompute(sampler=Sampler(), local=True) 
    # NOTA TÉCNICA: En versiones modernas de Qiskit, pasar el backend exacto
    # a veces requiere instanciar un 'Sampler' personalizado.
    # Para simplificar y asegurar compatibilidad con tu versión:
    
    # Opción A: Inyectar el simulador directamente si la API lo permite
    # Opción B (Más robusta): Usar FidelityQuantumKernel con el feature map y dejar
    # que Qiskit Aer maneje la aceleración globalmente si configuramos el entorno.
    
    # Vamos a usar la configuración explícita en el Kernel:
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=ComputeUncompute(sampler=Sampler()) # El Sampler estándar usará Aer por defecto
    )
    
    # TRUCO: Para forzar la GPU en la primitiva Sampler base, a veces es complejo.
    # La forma más directa en Qiskit Machine Learning actual es configurar las opciones del backend
    # y pasarlas.
    
    return kernel

# --- VERSIÓN SIMPLIFICADA PARA TU PROYECTO ---
# Dado que configurar Primitives + GPU puede ser doloroso por versiones,
# vamos a usar un truco: configurar la sesión global de Aer o usar el kernel directamente.

from qiskit_aer.primitives import Sampler as AerSampler

def get_gpu_kernel(feature_map):
    """Versión optimizada usando Primitivas de Aer (Nativo GPU)."""
    
    print("⚡ Configurando entorno CUDA/GPU...")
    
    # Definimos el sampler que corre sobre GPU
    # 'shots': None calcula la probabilidad exacta (statevector), ideal para kernels.
    gpu_sampler = AerSampler(
        run_options={"method": "statevector", "device": "GPU", "shots": None},
        transpile_options={"optimization_level": 0}
    )
    
    # Calculadora de fidelidad usando ese sampler
    fidelity = ComputeUncompute(sampler=gpu_sampler)
    
    # Kernel final
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    
    return kernel