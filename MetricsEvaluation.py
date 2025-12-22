import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MetricsEvaluator:
    def __init__(self, y_true, y_pred, y_scores=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_scores = np.array(y_scores) if y_scores is not None else None

        # Definiamo le classi
        self.pos_label = 4
        self.neg_label = 2

        # Inizializziamo i valori a None
        self.tp = self.tn = self.fp = self.fn = 0

    def get_metrics(self):
        """Calcola True Positive, True Negative, False Positive, False Negative"""
        self.tp = np.sum((self.y_true == self.pos_label) & (self.y_pred == self.pos_label))
        self.tn = np.sum((self.y_true == self.neg_label) & (self.y_pred == self.neg_label))
        self.fp = np.sum((self.y_true == self.neg_label) & (self.y_pred == self.pos_label))
        self.fn = np.sum((self.y_true == self.pos_label) & (self.y_pred == self.neg_label))

        accuracy = (self.tp + self.tn) / len(self.y_true) if len(self.y_true) > 0 else 0
        error_rate = 1 - accuracy

        sensitivity = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        g_mean = np.sqrt(sensitivity * specificity)

        return {
            "Accuracy": accuracy,
            "Error Rate": error_rate,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Geometric Mean": g_mean
        }

    def plot_confusion_matrix(self):
        """Plotta la matrice di confusione"""
        # Assicuriamoci che i conteggi siano aggiornati
        self.get_metrics()


        matrix_data = [
            [self.tn, self.fp],
            [self.fn, self.tp]
        ]

        fig, ax = plt.subplots(figsize=(6, 5))

        # Creiamo la visualizzazione a griglia
        for i in range(2):
            for j in range(2):
                val = matrix_data[i][j]
                # Disegna un box bianco con bordo nero per ogni cella
                ax.text(j, i, str(val), va='center', ha='center', fontsize=18, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=2.5'))

        # Configurazione assi
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predetto 2 (Neg)', 'Predetto 4 (Pos)'], fontsize=11)
        ax.set_yticklabels(['Reale 2 (Neg)', 'Reale 4 (Pos)'], fontsize=11)


        ax.set_ylim(1.5, -0.5)
        ax.set_xlim(-0.5, 1.5)

        plt.title('Matrice di Confusione', fontsize=14, pad=20)
        plt.ylabel('Classe Effettiva', fontsize=12)
        plt.xlabel('Classe Predetta', fontsize=12)

        # Rimuove la griglia e i bordi del grafico di sfondo
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which='both', length=0)

        plt.tight_layout()
        plt.show()
