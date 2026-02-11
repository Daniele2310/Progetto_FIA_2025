import numpy as np
import pandas as pd
from typing import Tuple

from Model_Development import KNN_Classifier
from holdout import holdout
from MetricsEvaluation import MetricsEvaluator


class KNN_Optimal:
    """
    Classe per la ricerca del k ottimale
    """

    def __init__(self, X_train: pd.DataFrame, Y_train: pd.Series, k_range: range, splits):
        """
        X: Features del dataset
        Y: Target del dataset
        k_range: Range di k da testare
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.k_range = k_range
        self.results = {}
        self.splits = splits

    def K_Holdout(self, test_size, random_state) -> Tuple[int, float]:
        """
        Trova il k ottimale usando Holdout Validation
        """
        # Eseguo lo split una sola volta
        X_sub_train, X_val, Y_sub_train, Y_val = holdout(
            self.X_train,
            self.Y_train,
            test_size,
            random_state
        )

        best_k = -1
        best_acc = 0.0

        # Itero su tutti i k nel range
        for k in self.k_range:
            knn = KNN_Classifier(K=k)
            knn.fit(X_sub_train, Y_sub_train)

            prediction = knn.predict(X_val)
            y_proba = knn.predict_proba(X_val)

            # Calcolo accuracy
            evaluator = MetricsEvaluator(Y_val, prediction, y_proba, pos_label=4)
            metriche = evaluator.get_metrics()
            acc = metriche['Accuracy']

            if acc > best_acc:
                best_acc = acc
                best_k = k

        self.results['Holdout'] = {'K': best_k, 'Accuracy': best_acc}
        return best_k, best_acc

    def K_random_subsampling(self, test_size, splits, random_state) -> Tuple[int, float]:
        """
        Trova il k ottimale usando Random Subsampling (media su n_iter split)
        """
        k_mean_performances = {}
        for k in self.k_range:
            accuracies = []

            # Per ogni K, testiamo su tutti gli split passati in ingresso
            for split in splits:
                X_train_split, _, Y_train_split, _ = split
                # Dividiamo ogni splits in validation e sub_train
                X_sub_train, X_val, Y_sub_train, Y_val = holdout(
                    X_train_split,
                    Y_train_split,
                    test_size,
                    random_state
                )

                knn = KNN_Classifier(K=k)
                knn.fit(X_sub_train, Y_sub_train)
                prediction = knn.predict(X_val)
                y_proba = knn.predict_proba(X_val)

                evaluator = MetricsEvaluator(Y_val, prediction, y_proba, pos_label=4)
                metriche = evaluator.get_metrics()
                acc = metriche['Accuracy']
                accuracies.append(acc)

            # Calcoliamo la media delle performance per questo specifico K
            k_mean_performances[k] = np.mean(accuracies)

        # Selezioniamo il K che ha la media più alta
        best_k = max(k_mean_performances, key=k_mean_performances.get)
        best_acc = k_mean_performances[best_k]

        self.results['Random Subsampling'] = {'K': best_k, 'Accuracy': best_acc}
        return best_k, best_acc

    def K_bootstrap(self, n_boot, random_state, splits, test_size) -> Tuple[int, float]:
        """
        Trova il k ottimale usando Bootstrap
        """
        k_performances = {}
        for k in self.k_range:
            accuracies = []

            for split in splits:
                X_train_split, _, Y_train_split, _ = split

                X_sub_train, X_val, Y_sub_train, Y_val = holdout(
                    X_train_split,
                    Y_train_split,
                    test_size,
                    random_state
                )

                knn = KNN_Classifier(K=k)
                knn.fit(X_sub_train, Y_sub_train)
                prediction = knn.predict(X_val)
                y_proba = knn.predict_proba(X_val)

                evaluator = MetricsEvaluator(Y_val, prediction, y_proba, pos_label=4)
                metriche = evaluator.get_metrics()
                acc = metriche['Accuracy']
                accuracies.append(acc)

            avg_acc = np.mean(accuracies)
            k_performances[k] = avg_acc

        best_k = max(k_performances, key=k_performances.get)
        best_acc = k_performances[best_k]

        self.results['Bootstrap'] = {'K': best_k, 'Accuracy': best_acc}
        return best_k, best_acc



