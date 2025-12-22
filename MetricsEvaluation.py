import numpy as np
import matplotlib.pyplot as plt
from holdout import holdout
from random_subsampling import random_subsampling


class MetricsEvaluator:
    def __init__(self, Y_true, Y_pred, Y_scores=None):

        # Convertiamo in array numpy, gestendo anche pandas Series.
        # Holdout restituisce pandas Series (usando .iloc[]),
        # mentre le operazioni di calcolo delle metriche richiedono numpy array.
        
        # .values è un attributo di pandas Series che restituisce i dati come numpy array.
        # hasattr() controlla se l'oggetto ha quell'attributo (cioè se è un pandas Series).
        if hasattr(Y_true, 'values'):
            self.Y_true = Y_true.values
        else:
            self.Y_true = np.array(Y_true)
            
        if hasattr(Y_pred, 'values'):
            self.Y_pred = Y_pred.values
        else:
            self.Y_pred = np.array(Y_pred)
            
        self.Y_scores = np.array(Y_scores) if Y_scores is not None else None

        # Definiamo le classi
        self.pos_label = 4
        self.neg_label = 2

        # Inizializziamo i valori a None
        self.tp = self.tn = self.fp = self.fn = 0

    def get_metrics(self):
        """Calcola True Positive, True Negative, False Positive, False Negative"""
        self.tp = np.sum((self.Y_true == self.pos_label) & (self.Y_pred == self.pos_label))
        self.tn = np.sum((self.Y_true == self.neg_label) & (self.Y_pred == self.neg_label))
        self.fp = np.sum((self.Y_true == self.neg_label) & (self.Y_pred == self.pos_label))
        self.fn = np.sum((self.Y_true == self.pos_label) & (self.Y_pred == self.neg_label))

        accuracy = (self.tp + self.tn) / len(self.Y_true) if len(self.Y_true) > 0 else 0
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
        self.get_metrics()

        matrix_data = [
            [self.tn, self.fp],
            [self.fn, self.tp]
        ]

        fig, ax = plt.subplots(figsize=(6, 5))

        for i in range(2):
            for j in range(2):
                val = matrix_data[i][j]
                ax.text(j, i, str(val), va='center', ha='center', fontsize=18, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=2.5'))

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predetto 2 (Neg)', 'Predetto 4 (Pos)'], fontsize=11)
        ax.set_yticklabels(['Reale 2 (Neg)', 'Reale 4 (Pos)'], fontsize=11)

        ax.set_ylim(1.5, -0.5)
        ax.set_xlim(-0.5, 1.5)

        plt.title('Matrice di Confusione', fontsize=14, pad=20)
        plt.ylabel('Classe Effettiva', fontsize=12)
        plt.xlabel('Classe Predetta', fontsize=12)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which='both', length=0)

        plt.tight_layout()
        plt.show()