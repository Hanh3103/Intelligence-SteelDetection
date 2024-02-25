# T-distributed Stochastic Neighbor Embedding (t-SNE)

Die Idee von t-SNE besteht darin, eine Darstellung hochdimensionaler Daten in einem niedrigdimensionalen Raum zu erzeugen, während die Struktur der Daten erhalten bleibt. Dies geschieht durch die Definition einer Wahrscheinlichkeitsverteilung im hochdimensionalen Raum, die angibt, wie wahrscheinlich es ist, dass zwei zufällig ausgewählte Punkte im Raum direkte Nachbarn sind.

Dieser Prozess ermöglicht es, komplexe Strukturen in den Daten zu bewahren und gleichzeitig eine kompakte Darstellung zu erzeugen, die für Visualisierungszwecke oder zur Dimensionsreduktion verwendet werden kann.

Der Algorithmus ist in der Datei `MLModels/unsupervised/cluster_Algorithm.py` implementiert. Es wird die Implementierung von `sklearn.manifold` genutzt. Durch den Befehl `tsne_algorithm(data, perplexity, steps, mode)` wird die Funktion aufgerufen.

Die Funktion nutzt zu Beginn eine logarithmische Skalierung. Diese könnte optional sein. Danach wird, je nach zuvor getroffener Auswahl, der Algorithmus ein Ergebnis im zweidimensionalen oder dreidimensionalen Raum liefern.

## Parameter:

- `DataFrame data`: Der zu verwendende Datensatz muss ein Pandas DataFrame sein. Die Spalten sollten dem Muster des zuvor erwähnten Scalars entsprechen, falls man diesen verwenden möchte.
- `Float perplexity`: Dieser Parameter gibt an, mit wie vielen Datenpunkten sich ein einzelner vergleicht. Je höher der Wert ist, desto klarer sind meistens die Trennungen der Cluster. Dieser Parameter hat einen sehr hohen Einfluss auf das Ergebnis.
- `Integer steps`: Die Anzahl an Iterationen, die der Algorithmus durchläuft.
- `String mode`: Eine Auswahl zwischen '2D' und '3D' ist möglich, da man diese Ergebnisse plotten kann.

## Output:

Eine Liste mit den Ergebnissen des Algorithmus, je nach Modusauswahl mit zwei oder drei Werten pro Datenpunkt.

# Affinity Propagation

Die Idee von Affinity Propagation basiert auf dem Austauschprinzip. Dabei werden Cluster durch den Austausch von Informationen zwischen Datenpaaren erzeugt, bis diese konvergieren. Der Algorithmus legt selbst die geschätzte Anzahl an Clustern fest und basiert sich dabei auf sogenannten Repräsentanten, die bei dem Nachrichtenaustausch ermittelt werden.

Die richtige Auswahl einer Distanzfunktion beeinflusst das Ergebnis stark. Die in dieser Implementierung genutzten Parameter haben nach zahlreichen Durchläufen auf den Testdaten die klarsten Cluster ergeben.

Der Algorithmus ist in der Datei `MLModels/unsupervised/cluster_Algorithm.py` implementiert. Es wird die Implementierung von `sklearn.cluster` genutzt. Durch den Befehl `affinity_propagation(data)` wird die Funktion aufgerufen.

## Parameter:

- `DataFrame data`: Der zu verwendende Datensatz muss ein Pandas DataFrame sein. Die Spalten sollten dem Muster des zuvor erwähnten Scalars entsprechen, falls man diesen verwenden möchte.

Die restlichen Parameter sind festgelegt. Dabei spielt der `preference`-Parameter die größte Rolle. Nach einigen Analysen hat sich ergeben, dass die besten Ergebnisse mit `0.6*np.min(-pairwise_distances(df_scaled, metric='l1'))` erzielt wurden. Hier besteht natürlich die Gefahr für Overfitting. Bei neuen Daten müsste die Konstante eventuell angepasst werden.

## Output:

Das übergebene DataFrame mit einer zusätzlichen Spalte, die die Clusterzugehörigkeit der Daten als int Werte 0 bis k-1 enthält.

# Fuzzy C-Mean

Dieser Algorithmus wurde zwar auf die Daten angewendet, jedoch stellt der Output eine gewisse Schwierigkeit dar. Deswegen findet man eine Implementierung des Algorithmus, aber keine Einbindung in das endgültige Projekt.

Der Algorithmus bietet eine Alternative zu k-means und hat den Vorteil, dass das Prinzip des Soft Clustering realisiert wird. Dabei wird jedem Datenpunkt eine Wahrscheinlichkeit für jedes Cluster zugeteilt und keine eindeutige Zuordnung wie beim Hard Clustering.

Für diesen Algorithmus gibt es die `fuzzy_c_mean_fit(data, n_clusters)` sowie die `fuzzy_c_mean_predict(data, cntr)` Methode aus der `skfuzzy`-Bibliothek.
