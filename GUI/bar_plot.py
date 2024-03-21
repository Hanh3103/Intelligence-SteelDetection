from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import tkinter as tk


class BarPlot:

    def create_supervised_plot(self, predicted_steel_types, elemente, prediction_data):
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

        ax.set_xlabel('Prozentsatz der gemessenen Elemente')
        ax.set_title('Zusammensetzung und Identifikation der Messungen')

        # Setzen der y-Tick-Positionen und -Labels
        ax.set_yticks(ind)
        ax.set_yticklabels([f"{i+1}-{st}" for i, st in enumerate(predicted_steel_types)])

        ax.legend()
        return fig

    def create_unsupervised_plot(self, dataframe):
        # Identifizieren der 'Cluster'-Spalte und Überprüfen, ob die 'Grade'-Spalte existiert
        cluster_col = 'Cluster'
        grade_col = 'Grade'
        if grade_col in dataframe.columns:
            # 'Grade' existiert, also beide Spalten berücksichtigen
            sort_columns = [cluster_col, grade_col]
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.drop([cluster_col, grade_col])
            dataframe_sorted = dataframe.sort_values(by=sort_columns)
            labels = dataframe_sorted[cluster_col].astype(str) + " - " + dataframe_sorted[grade_col].astype(str)
        else:
            # Nur 'Cluster' berücksichtigen, da 'Grade' nicht existiert
            sort_columns = [cluster_col]
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.drop(cluster_col)
            dataframe_sorted = dataframe.sort_values(by=sort_columns)
            labels = dataframe_sorted[cluster_col].astype(str)

        # Erstellen des Plots
        fig, ax = plt.subplots(figsize=(10, len(dataframe_sorted) * 0.4), dpi=100)

        # Farben für die verschiedenen Elemente (numerischen Spalten)
        color_map = plt.get_cmap('tab20')
        colors = color_map(np.linspace(0, 1, len(numeric_cols)))

        # Initialisiere die Startposition für das Stapeln der Balken
        left = np.zeros(len(dataframe_sorted))

        for i, col in enumerate(numeric_cols):
            # Erzeuge Balken für jede Zeile
            ax.barh(range(len(dataframe_sorted)), dataframe_sorted[col], left=left, color=colors[i], label=col)
            # Update der Startposition für den nächsten Stapel
            left += dataframe_sorted[col].values

        # Setzen der y-Tick-Positionen und -Labels
        ax.set_yticks(range(len(dataframe_sorted)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Prozentsatz der gemessenen Elemente')
        ax.set_ylabel('Cluster' + (' - Grade' if grade_col in dataframe.columns else ''))
        ax.set_title('Zusammensetzung und Identifikation der Messungen')
        ax.legend(title='Elemente')

        plt.tight_layout()
        return fig


def save_dataframe_as_csv(dataframe):
    root = tk.Tk()
    root.withdraw()  # Wir wollen nicht das ganze Tkinter Fenster, nur den Dialog

    # Den FileChooser Dialog öffnen und den Dateipfad bekommen, wo die Datei gespeichert werden soll
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                                             title="Speichere klassifizierte Daten als CSV ab")

    if file_path:  # Überprüfen, ob der Benutzer nicht abgebrochen hat
        dataframe.to_csv(file_path,
                         index=False)  # Index=False, um zu verhindern, dass der DataFrame-Index mitgespeichert wird
    else:
        return