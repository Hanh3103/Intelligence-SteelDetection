import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer


class BarPlot:

    def create_supervised_Plot(self, predicted_steel_types, elemente, prediction_data):
        df = pd.DataFrame(prediction_data)
        for col in df.columns:
            df[col] = df[col].str.replace(',', '.').astype(float)

        prediction_data = df.to_numpy()

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Erzeugen Positionen für die y-Achse
        ind = np.arange(len(predicted_steel_types))
        height = 0.35

        for i, elem in enumerate(elemente):
            # Erzeuge Balken iterativ
            left = np.sum(prediction_data[:, :i], axis=1) if i > 0 else 0
            ax.barh(ind, prediction_data[:, i], height, label=elem, left=left)

        ax.set_xlabel('Prozentsatz')
        ax.set_title('Prozentsatz der Elemente in verschiedenen Stahlsorten')

        # Setzen der y-Tick-Positionen und -Labels
        ax.set_yticks(ind)
        ax.set_yticklabels(predicted_steel_types)

        ax.legend()
        return fig

    def create_unsupervised_plot(self, data, k):
        df = pd.DataFrame(data)

        # Logarithmische Transformation anwenden
        scaler = FunctionTransformer(np.log1p, validate=True)
        df_scaled = scaler.fit_transform(df)

        # PCA anwenden, um Dimensionen auf 3 zu reduzieren
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(df_scaled)

        # Anwenden von K-Means-Clustering mit dem übergebenen k
        kmeans = KMeans(n_clusters=k, n_init=10)
        clusters = kmeans.fit_predict(reduced_data)

        # Erstellen einer Figur für das 3D-Scatter-Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=clusters, cmap='viridis', marker='o')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'K-means Clustering with k={k}')

        plt.tight_layout()

        # Rückgabe der Figur
        return fig

