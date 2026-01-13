"""
Quantum Backend Configuration.
Gestiona el acceso al simulador AER para CPU/GPU usando primitivas V2.
Aplica transpilación forzosa para evitar errores de instrucciones desconocidas.
"""
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute

def get_gpu_kernel(feature_map):
    """
    Configura el kernel usando qiskit-aer SamplerV2.
    Transpila el circuito y activa GPU si es posible.
    """
    print("⚙️ Configurando Backend de Simulación (Aer V2)...")
    
    # 1. TRANSPILACIÓN PREVENTIVA (CRÍTICO)
    # Convertimos el "ZZFeatureMap" abstracto en compuertas físicas (CX, RZ, H)
    # que el simulador Aer sí entiende.
    print("   -> Transpilando circuito para el simulador...")
    backend_temp = AerSimulator()
    transpiled_map = transpile(feature_map, backend_temp)
    
    # Instanciamos el Sampler V2
    sampler = SamplerV2()
    
    # INTENTO 1: Configurar GPU
    try:
        print("   -> Buscando GPU NVIDIA...")
        
        # Opciones para GPU
        sampler.options.backend_options = {
            "method": "statevector",
            "device": "GPU"
        }
        
        # Prueba de fuego con el circuito YA TRANSPILADO
        pub = (transpiled_map, [0] * transpiled_map.num_parameters)
        
        # Ejecutamos (esto lanzará error si no hay CUDA)
        job = sampler.run([pub])
        result = job.result()
        
        print("   🟢 ÉXITO: GPU NVIDIA activada.")

    except Exception as e:
        # INTENTO 2: Fallback a CPU
        print(f"   ⚠️ GPU no disponible ({e}).")
        print("   🔵 Usando CPU (Intel Standard).")
        
        # Revertimos a CPU
        sampler.options.backend_options = {
            "method": "statevector",
            "device": "CPU"
        }

    # Conectar el Sampler V2 a la calculadora de Fidelidad
    fidelity = ComputeUncompute(sampler=sampler)
    
    # Crear el Kernel USANDO EL MAPA TRANSPILADO
    kernel = FidelityQuantumKernel(feature_map=transpiled_map, fidelity=fidelity)
    
    return kernel