import numpy as np


class MetricsEvaluator:
    def __init__(self, y_true, y_pred, y_scores=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_scores = np.array(y_scores) if y_scores is not None else None

        # Definiamo le classi
        self.pos_label = 4
        self.neg_label = 2


    def get_metrics(self):
        """Calcola True Positive, True Negative, False Positive, False Negative"""
        self.tp = np.sum((self.y_true == self.pos_label) & (self.y_pred == self.pos_label))
        self.tn = np.sum((self.y_true == self.neg_label) & (self.y_pred == self.neg_label))
        self.fp = np.sum((self.y_true == self.neg_label) & (self.y_pred == self.pos_label))
        self.fn = np.sum((self.y_true == self.pos_label) & (self.y_pred == self.neg_label))

        # Assicuriamoci che i valori siano aggiornati
        accuracy = (self.tp + self.tn) / len(self.y_true)
        error_rate = 1 - accuracy

        # Gestione divisione per zero per sensibilità e specificità
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
    