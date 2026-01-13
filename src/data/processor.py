"""
Data Processor Module.

Encargado de la ingesta, limpieza, balanceo y transformación (PCA)
de los datos financieros para ser codificados en estados cuánticos.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os

class CryptoFraudData:
    """Manejador de datos para detección de fraude."""

    def __init__(self, n_dim: int = 2, sample_size: int = 200):
        """
        Args:
            n_dim (int): Número de dimensiones a conservar (qubits necesarios).
            sample_size (int): Número total de muestras a usar (limitado por velocidad de simulación).
        """
        self.n_dim = n_dim
        self.sample_size = sample_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_process(self, filepath: str = 'data/raw/creditcard.csv'):
        """Carga datos reales o genera sintéticos si el archivo no existe."""
        
        # 1. Ingesta
        if os.path.exists(filepath):
            print(f"📂 Cargando dataset real desde {filepath}...")
            df = pd.read_csv(filepath)
            
            # Balanceo de clases (Undersampling)
            fraud = df[df.Class == 1]
            legal = df[df.Class == 0].sample(n=len(fraud), random_state=42)
            data = pd.concat([fraud, legal]).sample(frac=1, random_state=42)
            
            X = data.drop(['Class'], axis=1).values
            y = data['Class'].values
        else:
            print("⚠️ No se encontró dataset real. Generando datos sintéticos (Mock)...")
            # Generamos datos que imitan transacciones financieras
            X, y = make_classification(
                n_samples=self.sample_size * 2, # Un poco más para tener margen
                n_features=30, 
                n_informative=self.n_dim + 1, # Un poco de señal real
                n_redundant=0, 
                weights=[0.5, 0.5], # Balanceado para entrenamiento
                random_state=42
            )

        # Recortar al tamaño de muestra deseado (Simulación cuántica es lenta)
        X = X[:self.sample_size]
        y = y[:self.sample_size]

        # 2. Pipeline Matemático
        print("🧮 Aplicando PCA y Re-escalado...")
        
        # A. Estandarización (Media 0, Varianza 1) - Crucial para PCA
        std_scaler = StandardScaler()
        X_std = std_scaler.fit_transform(X)

        # B. PCA (Proyección a subespacio de n_dim)
        pca = PCA(n_components=self.n_dim)
        X_pca = pca.fit_transform(X_std)

        # C. Escalado MinMax (Mapeo a rango [-1, 1] para rotaciones cuánticas)
        # Esto es vital: los valores se convierten en coeficientes de rotación.
        minmax = MinMaxScaler(feature_range=(-1, 1))
        X_final = minmax.fit_transform(X_pca)

        # 3. Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_final, y, test_size=0.3, random_state=42
        )
        
        print(f"✅ Datos listos. Shape entrenamiento: {self.X_train.shape}")
        return self

# Bloque de prueba
if __name__ == "__main__":
    processor = CryptoFraudData(n_dim=2, sample_size=100)
    processor.load_and_process()
    print("Muestra de datos (primeros 3 vectores):")
    print(processor.X_train[:3])