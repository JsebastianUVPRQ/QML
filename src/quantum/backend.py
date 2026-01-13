"""
Quantum Backend Configuration.
Gestiona el acceso al simulador AER para CPU/GPU usando primitivas nativas.
"""
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute

def get_gpu_kernel(feature_map):
    """
    Configura el kernel usando qiskit-aer.
    Intenta activar GPU, si falla, usa CPU silenciosamente.
    """
    print("⚙️ Configurando Backend de Simulación (Aer)...")
    
    try:
        # INTENTO 1: Configurar Sampler con GPU
        # run_options={"device": "GPU"} es la clave para qiskit-aer
        print("   -> Buscando GPU NVIDIA...")
        
        sampler = AerSampler(
            run_options={"method": "statevector", "device": "GPU", "shots": None},
            transpile_options={"optimization_level": 0}
        )
        
        # Pequeña prueba de fuego para ver si la GPU responde sin error
        # (Esto lanzará excepción si no hay drivers CUDA)
        dummy_circ = feature_map.bind_parameters([0.1]*feature_map.num_parameters)
        job = sampler.run([dummy_circ])
        _ = job.result() 
        
        print("   🟢 ÉXITO: GPU NVIDIA activada.")
        
    except Exception as e:
        # INTENTO 2: Fallback a CPU
        # El error suele ser "AerError: GPU not available" o similar
        print(f"   ⚠️ GPU no disponible o mal configurada ({e}).")
        print("   🔵 Usando CPU (Intel).")
        
        sampler = AerSampler(
            run_options={"method": "statevector", "device": "CPU", "shots": None}
        )

    # Conectar el Sampler a la calculadora de Fidelidad
    fidelity = ComputeUncompute(sampler=sampler)
    
    # Crear el Kernel
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    
    return kernel