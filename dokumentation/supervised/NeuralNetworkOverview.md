
# DeepLearning1
Implemented by Julius

Diese Klasse realisiert das SteelModelInterface mit einem tiefen neuronalen Netz. Hierfür wird tensorflow.keras genutzt, da es ein bewährtes Modul zur Erstellung tiefer neuronaler Netze ist..

## Aufbau des Netzes:

Das neuronale Netz besteht aus 5 Schichten mit jeweils 248, 124, 64 und 32 Neuronen in der Input- und Hidden-Schicht. Die Aktivierungsfunktion bei den ersten 4 Schichten ist ReLu, welche eine Standardwahl darstellt und keinen Anlass zum Wechsel gegeben hat. Die Aktivierungsfunktion im Output Layer ist wie bei Mehrfachklassifizierung typisch Softmax. Außerdem gibt es eine Dropout-Schicht und einen L2-Regularisierer in Schicht drei. Diese haben bei durchgeführten Kreuzvalidierungsexperimenten die Ergebnisse für die Testdaten verbessert und damit effektiv Overfitting reduziert. Die Anzahl der Schichten sowie aller übriger Hyperparameter wurde rein empirisch ermittelt. Die Hyperparameter wurden per Hand verstellt und Schichten hinzugefügt, bis es bei den Tests keine signifikanten Abweichungen mehr in der accuracy, loss und f1-score gab. Im Durchschnitt erreicht das Netz nun bei den zur Verfügung gestellten Daten einen f1-score von 0.94. Dieses Ergebnis haben wir als ausreichend bewertet. Die Metriken konvergieren bei ungefähr 800 Trainingsepochen mit dem Optimierer Adam.

## Datenaufbereitung:

Zur Datenaufbereitung haben wir aus allen uns zur Verfügung gestellten Daten eine Zusammenstellung in einer CSV-Datei erstellt. Wir haben von allen Daten die Schnittmenge der features genutzt, welche in den Elementen: C, Si, Mn, P ,S ,Cr ,Ni ,Al ,Cu besteht. Hierdurch erreicht unser Datensatz eine maximale Größe von 437 Trainingsbeispieln.
Die Konvertierung der Daten und Einteilung in Test- und Trainingsdaten erfolgt in der Methode `convert_data`. Diese kann mit einer beliebigen Anzahl von features umgehen und zieht die Labels der Zielvariablen und features direkt aus dem Datensatz. Hierdurch können weitere Stahlsorten oder Elemente ergänzt werden ohne den Quellcode anzupassen. Durch integer encoding werden die einzelnen Stahlsorten in Werte von eins bis N durchnummeriert. Die features sind bereits numerisch und normalisiert, da es sich um Anteile verschiedener chemischer Elemente handelt.

## Vorhersagen:

Die Vorhersage einer Instanz findet in der Funktion `predict` statt. Hier kann auch eine Liste mehrerer Instanzen übergeben werden. In der Rückgabe wird eine Liste mit den wahrscheinlichsten Labels und eine Liste mit den entsprechenden Wahrscheinlichkeiten zurückgegeben. Für die Rückgabe der Labels wird der in der Methode `convert_data` erstellte integer encoder oder ein geladener encoder genutzt.

## Testung:

Mit der Methode `test_model` kann eine Evaluierung des Modells durchgeführt werden. Da hierfür die Testdaten nötig sind, ist dies nicht bei einem geladenen, sondern nur frisch erstellten Modell möglich. In der Evaluation werden sparse categorical crossentropy loss, accuracy und f1-score für den gesamten Datensatz, als auch gesondert für die Testdaten ausgegeben. Hierdurch erhält der Nutzer direkt eine gute Einschätzung über die Güte seines Modells und kann durch die gesonderte Bewertung der Testdaten auch mögliches Overfitting abschätzen.
