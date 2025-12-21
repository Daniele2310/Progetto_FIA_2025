import numpy as np
from collections import Counter
import random

# Implementazione del K-NN classificatore che ha come iper-parametro K
class KNN_Classifier:

    def __init__(self, K):
        self.K = K

    def fit(self, X_train, Y_train):
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train).flatten()

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []

        # gestione il caso di un singolo campione
        if X_test.ndim == 1:
            X_test = [X_test]

        for samples in X_test:
            # Calcolo tutte le distanze (poi ne selezionerò solamente K)
            distances = np.linalg.norm(self.X_train - samples, axis=1)

            # Creo coppie (distanza, etichetta)
            neighbors_data = list(zip(distances, self.Y_train))

            # Ordino le coppie forzandolo a lavorare solamente sulla distanza, nel caso la
            #distanza sia uguale allora manterrà l'ordine con cui aveva letto i valori
            neighbors_data.sort(key=lambda x: x[0])

            # Analizzo i K vicini
            k_nearest = neighbors_data[:self.K]

            # Estraggo solamente le etichette dei K vicini (non considero le distanze)
            k_labels = [label for (dist, label) in k_nearest]

            # Conto quale etichetta appare più volte
            conteggio = Counter(k_labels)
            max_common = conteggio.most_common(1)[0][1]
            best_labels = [label for label, count in conteggio.items() if count == max_common]

            predictions.append(random.choice(best_labels))

        return np.array(predictions)

