import numpy as np
from matplotlib.figure import Figure
import pandas as pd


class BarPlot:

    def create_supervised_Plot(self, predicted_steel_types, elemente, prediction_data):
        df = pd.DataFrame(prediction_data)
        print(predicted_steel_types)
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
