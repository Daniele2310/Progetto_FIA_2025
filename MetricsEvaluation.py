import numpy as np
import matplotlib.pyplot as plt
from holdout import holdout
from random_subsampling import random_subsampling


class MetricsEvaluator:
    # Aggiunto parametro pos_label per rendere la classe flessibile (target 4 o 2)
    def __init__(self, Y_true, Y_pred, Y_scores=None, pos_label=4):

        # Converto in array numpy, gestendo anche pandas Series.
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

        # Definisco le classi in modo dinamico in base al parametro passato
        self.pos_label = pos_label

        # Cerco l'etichetta negativa.
        # Se nel vettore Y_true ci sono le classi [2, 4] e pos_label è 4, allora neg_label deve essere 2.
        unique_classes = np.unique(self.Y_true)
        if len(unique_classes) == 2:
            # Prende l'elemento che non è uguale a pos_label
            self.neg_label = unique_classes[unique_classes != self.pos_label][0]
        else:
            self.neg_label = 2 if self.pos_label == 4 else 4

        # Inizializzo i valori a None
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


        #Utilizziamo float() per migliorare la visualizzazione a schermo e round()
        # per limitare le cifre decimali
        return {
            "Accuracy": round(float(accuracy * 100), 2),
            "Error Rate": round(float(error_rate * 100), 2),
            "Sensitivity": round(float(sensitivity * 100), 2),
            "Specificity": round(float(specificity * 100), 2),
            "Geometric Mean": round(float(g_mean * 100), 2)
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

        # Le etichette degli assi ora usano self.pos_label e self.neg_label
        # per adattarsi a qualsiasi scelta (Target 4 o Target 2).
        ax.set_xticklabels([f'Pred {self.neg_label}', f'Pred {self.pos_label}'], fontsize=11)
        ax.set_yticklabels([f'Reale {self.neg_label}', f'Reale {self.pos_label}'], fontsize=11)

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

    def plot_roc_curve(self):
        if self.Y_scores is None:
            return 0.0

        # Per costruire la ROC, analizzo i campioni
        # partendo da quello con lo score (probabilità) più alto fino al più basso.
        # argsort restituisce gli indici ordinati crescenti, [::-1] li inverte.
        desc_indices = np.argsort(self.Y_scores)[::-1]

        y_true_sorted = self.Y_true[desc_indices]
        y_scores_sorted = self.Y_scores[desc_indices]

        # Totali di positivi (P) e negativi (N) reali nel dataset
        P_total = np.sum(self.Y_true == self.pos_label)
        N_total = np.sum(self.Y_true == self.neg_label)

        tpr_list = [0.0]
        fpr_list = [0.0]
        tp_count = 0
        fp_count = 0

        # Scansione delle soglie. Scendere nella lista equivale ad abbassare la soglia di decisione.
        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == self.pos_label:
                tp_count += 1  # Trovato un vero positivo, il grafico sale
            else:
                fp_count += 1  # Trovato un falso positivo, il grafico va a destra

            # Se due punti hanno lo stesso score identico, vanno processati insieme per evitare scalini falsi.
            if i == len(y_scores_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
                tpr_list.append(tp_count / P_total if P_total > 0 else 0)
                fpr_list.append(fp_count / N_total if N_total > 0 else 0)

        # Calcolo AUC usando la Regola del Trapezio
        # Si sommano le aree dei trapezi formati da ogni segmento della curva.
        auc = 0.0
        for i in range(1, len(fpr_list)):
            base = fpr_list[i] - fpr_list[i - 1]
            height = (tpr_list[i] + tpr_list[i - 1]) / 2
            auc += base * height

        plt.figure(figsize=(6, 5))
        plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

        return auc