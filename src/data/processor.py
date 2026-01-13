import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os

class CryptoFraudData:
    def __init__(self, n_dim: int = 2, sample_size: int = 200):
        self.n_dim = n_dim
        self.sample_size = sample_size
        self.X_train = None; self.X_test = None
        self.y_train = None; self.y_test = None

    def load_and_process(self, filepath: str = 'data/raw/creditcard.csv'):
        if os.path.exists(filepath):
            print(f"📂 Cargando dataset real desde {filepath}...")
            df = pd.read_csv(filepath)
            fraud = df[df.Class == 1]
            legal = df[df.Class == 0].sample(n=len(fraud), random_state=42)
            data = pd.concat([fraud, legal]).sample(frac=1, random_state=42)
            X = data.drop(['Class'], axis=1).values
            y = data['Class'].values
        else:
            print("⚠️ Dataset no encontrado. Generando Mock Data...")
            X, y = make_classification(n_samples=self.sample_size*2, n_features=30, n_informative=3)

        X = X[:self.sample_size]
        y = y[:self.sample_size]

        print("🧮 Aplicando PCA y Re-escalado...")
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        pca = PCA(n_components=self.n_dim)
        X_pca = pca.fit_transform(X_std)
        minmax = MinMaxScaler(feature_range=(-1, 1))
        X_final = minmax.fit_transform(X_pca)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_final, y, test_size=0.3, random_state=42
        )
        print(f"✅ Datos listos. Shape entrenamiento: {self.X_train.shape}")
        return self