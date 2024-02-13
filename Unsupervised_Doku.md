# Intelligente-Stahlerkennung
Dies ist Teil der Dokumentation des Fortgeschrittenen Praktikums "Intelligente Stahlerkennung" im Wintersemester 2023/24.
In diesem FoPra sollten die Studierenden supervised und unsupervised learning Methoden anwenden, um OES Messungen von 
verschiedenen Stahlproben ihrem zugehörigen Stahltyp zuzuordnen. Dies soll die Analyse von Stahlproben beschleunigen
und eine nicht invasive Methode zur Stahlerkennung ermöglichen.
Im folgendem werden die unsupervised learning Methoden vorgestellt, die im Rahmen des FoPra angewendet wurden.
## Unsuperivised Learning
Unsupervised Learning ist eine Methode des maschinellen Lernens, bei der ein Modell Muster in einem Datensatz erkennt, ohne dass die Daten gelabelt sind. Das bedeutet, dass das Modell selbstständig Muster in den Daten erkennen muss, ohne dass ihm gesagt wird, was es suchen soll. Unsupervised Learning wird häufig für die Datenanalyse verwendet, um Muster in großen Datenmengen zu erkennen. Es gibt verschiedene Methoden des unsupervised Learning, die im Rahmen des FoPra angewendet wurden.
### K-Means Clustering
K-Means Clustering ist ein Verfahren des unsupervised Learning, bei dem die Daten in k Gruppen (Cluster) eingeteilt werden. Dabei wird die Anzahl der Cluster k vorgegeben. Das Ziel des K-Means Clustering ist es, die Daten so auf die Cluster aufzuteilen, dass die Daten innerhalb eines Clusters möglichst ähnlich und die Daten zwischen den Clustern möglichst verschieden sind. Das Verfahren des K-Means Clustering wurde im Rahmen des FoPra angewendet, um die Stahlproben in verschiedene Klassen einzuteilen.

Die Methode kmeans ist in der Datei MLModels/unsupervised/cluster_Algorithm.py implementiert.
Die Methode kmeans nimmt ein DataFrame, die Anzahl der Cluster _k_ und optional einen Scaler als Parameter.
Die Methode erweitert das DataFrame um eine Spalte, die die Clusterzugehörigkeit der Daten als int Werte _0 bis k-1_ enthält.

`def kmeans(dataframe, k, scaler=None):`

Das Clustering benutzt den kmeans++ Algorithmus von sklearn.cluster.KMeans, um die Daten in k Cluster einzuteilen.
Hierbei werden nur die numerischen Spalten des DataFrames für das Clustering verwendet. 

Parameter:

DataFrame _dataframe_: Das übergebene DataFrame musss mindestens eine numerische Spalte enthalten.
Dies macht die Methode flexibel, da sie auf verschiedene Datensätzen mit unterschiedlich vielen Elementen angewendet werden kann.
Es sollte darauf geachtet werden, dass das übergebene DataFrame keine numerischen Spalten enthält, die nicht in das Clustering einbezogen werden sollen.

Integer _k_: Der Parameter _k_ kann frei gewählt werden (muss int und _k>=2_) und gibt die Anzahl der Cluster an, in die die Daten eingeteilt werden sollen.

Scaler _scaler_: Der Parameter _scaler_ ist optional und kann genutzt werden, um die Daten Spaltenweise zu skalieren.
Hierbei kann jeder Scaler aus sklearn.preprocessing verwendet werden. Ein Scaler kann genutzt werden, um die Daten zu normalisieren, sodass die Daten in den Spalten vergleichbar sind.
Wenn kein Scaler übergeben wird, werden die Daten nicht skaliert.

Output: das übergebene DataFrame mit einer zusätzlichen Spalte, die die Clusterzugehörigkeit der Daten als int Werte _0 bis k-1_ enthält.

Tests:
Es wurden Unit-Tests für die Methode kmeans implementiert, die in der Datei tests/unsupervised/test_kmeans.py zu finden sind.
Die Unit-Tests überprüfen, ob mit richtigen und falschen inputs richtig umgegangen wird und ob die Methode die erwarteten Ergebnisse liefert.
Des weiteren wurde die Methode kmeans auf zwei Datensätzen getestet, und die Ergebnisse visualisiert (Nicht in der Datei enthalten, siehe Jupyter Notebook <apan style="color:red">TODO: add reference</apan>).
Bei einem der Datensätze handelt es sich um einen Datensatz der die tatsächlichen Stahltypen enthält, um die Ergebnisse des K-Means Clustering zu validieren.

Verschiedene weitere Methoden und implementierte Funktionen, die im Rahmen des FoPra ausprobiert wurden, es aber aus verschiedenen Gründen nicht in die finale Version geschafft haben, sind in dem Jupyter Notebook <apan style="color:red">TODO: add reference</apan> zu finden.
 
### PCA
Principal Component Analysis (PCA) ist ein Verfahren des unsupervised Learning, bei dem die Dimensionalität der Daten reduziert wird. Das Ziel der PCA ist es, die Daten so zu transformieren, dass die Daten in den neuen Dimensionen möglichst viel Varianz aufweisen. Die PCA wird im Rahmen des FoPra angewendet, um die Daten zu visualisieren und die Dimensionalität der Daten zu reduzieren.

Die Methode pca ist in der Datei MLModels/unsupervised/clustering_Algorithm.py implementiert. 
Die Methode pca nimmt ein DataFrame und optional einen Scaler als Parameter.
Die Methode gibt einen Array zurück, der die explained variance ratio aller Elemente enthält.

`def pca(dataframe, scaler=None):`

Die Methode benutzt die PCA von sklearn.decomposition, um die Daten zu transformieren.
Hierbei werden nur die numerischen Spalten des DataFrames für die PCA verwendet.

Parameter:

DataFrame _dataframe_: Das übergebene DataFrame muss mindestens eine numerische Spalte enthalten.
Dies macht die Methode flexibel, da sie auf verschiedene Datensätzen mit unterschiedlich vielen Elementen angewendet werden kann.
Es sollte darauf geachtet werden, dass das übergebene DataFrame keine numerischen Spalten enthält, die nicht in die PCA einbezogen werden sollen.

Scaler _scaler_: Der Parameter _scaler_ ist optional und kann genutzt werden, um die Daten Spaltenweise zu skalieren.
Hierbei kann jeder Scaler aus sklearn.preprocessing verwendet werden. Ein Scaler kann genutzt werden, um die Daten zu normalisieren, sodass die Daten in den Spalten vergleichbar sind.
Wenn kein Scaler übergeben wird, werden die Daten nicht skaliert.

Output: Ein Array, das die explained variance ratio aller Elemente enthält.

Tests:
Es wurden Unit-Tests für die Methode pca implementiert, die in der Datei tests/unsupervised/test_pca.py zu finden sind.
Die Unit-Tests überprüfen, ob mit richtigen und falschen inputs richtig umgegangen wird und ob die Methode die erwarteten Ergebnisse liefert.
Des weiteren wurde die Methode pca auf zwei Datensätzen getestet, und die Ergebnisse visualisiert (Nicht in der Datei enthalten, siehe Jupyter Notebook <apan style="color:red">TODO: add reference</apan>).

Verschiedene weitere Methoden und implementierte Funktionen, die im Rahmen des FoPra ausprobiert wurden, es aber aus verschiedenen Gründen nicht in die finale Version geschafft haben, sind in dem Jupyter Notebook <apan style="color:red">TODO: add reference</apan> zu finden.


