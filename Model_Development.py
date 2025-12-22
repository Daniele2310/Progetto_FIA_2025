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

    def predict_proba(self, X_test, pos_label=4):
        """
        Calcola la probabilità che i campioni di test appartengano alla classe positiva (pos_label).
        A differenza di 'predict' che restituisce la classe, questo restituisce un numero tra 0.0 e 1.0
        """
        X_test = np.array(X_test)
        scores = []

        if X_test.ndim == 1:
            X_test = [X_test]

        for sample in X_test:
            dists = np.linalg.norm(self.X_train - sample, axis=1)

            # Creo una lista di coppie (distanza, etichetta_reale).
            # La funzione 'zip' accoppia la distanza calcolata con la vera classe del punto di train.
            dist_label_pairs = zip(dists, self.Y_train)

            # Ordino le coppie in base alla distanza
            neighbors = sorted(dist_label_pairs, key=lambda x: x[0])[:self.K]

            labels = [n[1] for n in neighbors]

            # Conto quanti dei vicini appartengono alla classe "Positiva" (4 - Maligno)
            positive_votes = labels.count(pos_label)

            # La probabilità è data dalla frequenza relativa
            scores.append(positive_votes / self.K)

        # Restituisco un array di probabilità
        return np.array(scores)
