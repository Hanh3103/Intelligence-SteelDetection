# Bar Plot Visualization Tool

Dieses Tool ermöglicht die Visualisierung von Daten in Form von Balkendiagrammen. 

## Features

- **Supervised Plot Erstellung**: Visualisiert die Zusammensetzung und Identifikation der Messungen basierend auf vorhergesagten Stahltypen und Elementen.
- **Unsupervised Plot Erstellung**: Visualisiert die Zusammensetzung und Identifikation der Messungen basierend auf Clustern und, falls vorhanden, Grade-Klassifikationen.
- **Speichern von Daten als CSV**: Ermöglicht das Speichern der klassifizierten Daten in einer CSV-Datei.

## Klassen und Methoden

### BarPlot

#### Methoden

- `create_supervised_plot(self, predicted_steel_types, elemente, prediction_data)`: Erstellt ein Balkendiagramm für supervisierte Vorhersagedaten.
    - `predicted_steel_types`: Liste der vorhergesagten Stahltypen.
    - `elemente`: Liste der Elemente, die in den Messungen identifiziert wurden.
    - `prediction_data`: Datenframe mit den Vorhersagedaten.

- `create_unsupervised_plot(self, dataframe)`: Erstellt ein Balkendiagramm für unsupervisierte Daten.
    - `dataframe`: Datenframe, der die zu visualisierenden Daten enthält.

### save_dataframe_as_csv

- `save_dataframe_as_csv(dataframe)`: Speichert einen gegebenen Datenframe als CSV-Datei.
    - `dataframe`: Der zu speichernde Datenframe.


## Future Work

In the future, the `create_supervised_plot` and `create_unsupervised_plot` methods could be merged if the return of supervised/unsupervised algorithms is changed to a uniform data type, such as a Pandas DataFrame. This would simplify the codebase and increase flexibility in processing the outputs of different algorithms. Additionally, further plot options such as scatter plots or tabular representations could be created. However, implementing such visualizations with the current capabilities of Tkinter proves to be unfavorable. Moreover, if the return of the supervised algorithms is adjusted to a Pandas DataFrame, an identifier (e.g., a column in the DataFrame) should be included, which allows the predictions to be uniquely assigned to the original data (see unsupervised).
