import unittest
from unittest.mock import patch
import numpy as np

from Data_Preprocessing import Data_Loader
from Model_Development import KNN_Classifier
from holdout import holdout
from random_subsampling import random_subsampling
from Bootstrap import bootstrap

FILENAME = "Dataset_Tumori.csv"

CLASSES_COL = "classtype_v1"

FEATURES_LIST = [
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bare Nuclei',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses'
    ]


class Test_Data_Loader_Cleaning(unittest.TestCase):
    """
    Testa che il caricamento e la pulizia dei dati avvengano correttamente.
    """

    def setUp(self):
        # Questo metodo viene eseguito prima di ogni test

        # Patch sostituisce la print vera con una finta che non fa nulla
        with patch('builtins.print'):
            # Qui dentro le print vengono ignorate totalmente
            self.loader = Data_Loader(FILENAME, FEATURES_LIST, CLASSES_COL)
            self.loader.load_dataset()
            self.loader.features_cleaning_and_extraction()

        self.X = self.loader.X
        self.Y = self.loader.Y

    def test_loader_popola_variabili(self):
        """Verifica che X e Y siano popolati e abbiano lo stesso numero di righe."""

        self.assertFalse(self.X.empty, "Errore: X è vuoto.")
        self.assertFalse(self.Y.empty, "Errore: Y è vuoto.")
        self.assertEqual(len(self.X), len(self.Y), "Errore: X e Y hanno lunghezze diverse.")

    def test_regola_valori_maggiori_dieci(self):
        """
        Verifica che i valori > 10 siano stati rimossi.
        """
        matrice_dati = self.X.values
        # Cerchiamo il valore massimo nell'intero dataset (ignorando i Nan)
        valore_massimo = np.nanmax(matrice_dati)

        self.assertLessEqual(valore_massimo, 10,
                             f"Errore pulizia: Trovato valore {valore_massimo} che è > 10.")

    def test_assenza_nan_finali(self):
        """Verifica che dopo l'imputazione con la mediana non ci siano NaN residui."""
        conteggio_nan = self.X.isnull().sum().sum()
        self.assertEqual(conteggio_nan, 0, f"Errore: Ci sono ancora {conteggio_nan} valori mancanti (NaN).")

    def test_rimozione_colonna_id(self):
        """
        Verifica che la colonna ('Sample code number')
        sia stata rimossa correttamente dal dataset finale X.
        """
        colonna_id = 'Sample code number'
        # La colonna ID non deve essere presente nelle colonne di X
        self.assertNotIn(colonna_id, self.X.columns,f"Errore: La colonna '{colonna_id}' non è stata rimossa dal dataset.")

        colonne_reali = set(self.X.columns)
        colonne_attese = set(FEATURES_LIST)

        # Le colonne presenti devono essere quelle specificate in FEATURES_LIST
        # Controlliamo che le colonne reali siano un sottoinsieme o uguali a quelle attese
        self.assertTrue(colonne_reali.issubset(colonne_attese),
                        f"Trovate colonne impreviste: {colonne_reali - colonne_attese}")




# ===========TEST HOLDOUT===========

class Test_Holdout(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(100).reshape(50, 2)
        self.Y = np.arange(50)
        self.test_size = 0.2
        self.seed = 42

    def test_holdout_dimensioni_corrette(self):
        X_train, X_test, Y_train, Y_test = holdout(
            self.X, self.Y, self.test_size, self.seed
        )

        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.Y))

    def test_holdout_percentuale_test(self):
        X_train, X_test, _, _ = holdout(
            self.X, self.Y, self.test_size, self.seed
        )

        expected_test_size = int(len(self.X) * self.test_size)
        self.assertEqual(len(X_test), expected_test_size)


# ===========TEST RANDOMSUBSAMPLING===========
class Test_RandomSubsampling(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(100).reshape(50, 2)
        self.Y = np.arange(50)
        self.test_size = 0.3
        self.n_iter = 5
        self.seed = 42

    def test_numero_split(self):
        splits = random_subsampling(
            self.X, self.Y, self.test_size, self.n_iter, self.seed
        )

        self.assertEqual(len(splits), self.n_iter)

    def test_split_dimensioni(self):
        splits = random_subsampling(
            self.X, self.Y, self.test_size, self.n_iter, self.seed
        )

        for X_train, X_test, Y_train, Y_test in splits:
            self.assertEqual(len(X_train) + len(X_test), len(self.X))
            self.assertEqual(len(Y_train) + len(Y_test), len(self.Y))


# ===========TEST BOOTSTRAP===========
class Test_Bootstrap(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(100).reshape(50, 2)
        self.Y = np.arange(50)
        self.k = 5
        self.seed = 42

    def test_numero_bootstrap(self):
        splits = bootstrap(self.X, self.Y, self.k, self.seed)
        self.assertEqual(len(splits), self.k)

    def test_dimensione_training_set(self):
        splits = bootstrap(self.X, self.Y, self.k, self.seed)

        for X_train, _, Y_train, _ in splits:
            self.assertEqual(len(X_train), len(self.X))
            self.assertEqual(len(Y_train), len(self.Y))
    
    
    























































































































































































































from MetricsEvaluation import MetricsEvaluator  # Importazione della classe


class Test_Metrics_Evaluation(unittest.TestCase):
    """
    Test per verificare la correttezza dei calcoli delle metriche di performance
    """

    def setUp(self):
        """
        Inizializza un caso di test con valori noti:
        - 2 campioni reali positivi (4) e 2 negativi (2)
        - Predizioni: 1 True Positive, 2 True Negative, 0 False Positive, 1 False Negative
        """
        self.y_true = np.array([4, 4, 2, 2])
        self.y_pred = np.array([4, 2, 2, 2])
        # Risultati attesi: Accuratezza 75%, Sensibilità 50%, Specificità 100%

        self.evaluator = MetricsEvaluator(self.y_true, self.y_pred, pos_label=4)

    def test_calcolo_accuratezza(self):
        """Verifica che l'accuratezza sia (TP+TN)/Totale = 3/4 = 75.0%."""
        metrics = self.evaluator.get_metrics()
        self.assertEqual(metrics["Accuracy"], 75.0)

    def test_calcolo_sensibilita(self):
        """Verifica la sensibilità (Recall): TP/(TP+FN) = 1/2 = 50.0%."""
        metrics = self.evaluator.get_metrics()
        self.assertEqual(metrics["Sensitivity"], 50.0)

    def test_calcolo_specificita(self):
        """Verifica la specificità: TN/(TN+FP) = 2/2 = 100.0%."""
        metrics = self.evaluator.get_metrics()
        self.assertEqual(metrics["Specificity"], 100.0)

    def test_divisione_per_zero_metriche(self):
        """Verifica che l'evaluator gestisca array vuoti senza crashare."""
        empty_eval = MetricsEvaluator(np.array([]), np.array([]))
        metrics = empty_eval.get_metrics()
        self.assertEqual(metrics["Accuracy"], 0.0) # Dovrebbe restituire 0 invece di ZeroDivisionError

    @patch('matplotlib.pyplot.show')
    def test_auc_ranking_perfetto(self, mock_show):
        """
        Testa il calcolo dell'AUC utilizzando un Mock per isolare Matplotlib.
        In un caso di ranking perfetto, l'AUC deve essere 1.0.
        """
        y_scores = np.array([0.9, 0.8, 0.1, 0.2])
        eval_auc = MetricsEvaluator(self.y_true, self.y_pred, Y_scores=y_scores, pos_label=4)

        auc = eval_auc.plot_roc_curve()

        # Verifica che il grafico non sia apparso a video e l'AUC sia corretta
        self.assertTrue(mock_show.called)
        self.assertEqual(auc, 1.0)

class Test_KNN(unittest.TestCase):
    def setUp(self):
        self.knn = KNN_Classifier(K=3)

    def test_knn_fit_input_unidimensionale(self):
        """Verifica che il KNN gestisca correttamente Y_train se passato come colonna singola."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        Y_train = np.array([[2], [4], [2]]) # Formato (3, 1) invece di (3,)
        self.knn.fit(X_train, Y_train)
        self.assertEqual(self.knn.Y_train.ndim, 1) # Dovrebbe essere flatten() nel fit

    def test_knn_predict_singolo_campione(self):
        """Verifica che predict funzioni anche passando un singolo vettore invece di una matrice."""
        X_train = np.array([[1, 1], [10, 10]])
        Y_train = np.array([2, 4])
        self.knn.K = 1
        self.knn.fit(X_train, Y_train)
        pred = self.knn.predict(np.array([1.1, 1.1])) # Input shape (2,)
        self.assertEqual(len(pred), 1)

    def test_knn_pareggio_voti(self):
        """Verifica il comportamento del KNN in caso di pareggio (K=2)."""
        knn = KNN_Classifier(K=2)
        # Due punti di training: uno classe 2 e uno classe 4
        X_train = np.array([[1, 1], [2, 2]])
        Y_train = np.array([2, 4])
        knn.fit(X_train, Y_train)
        
        # Punto di test esattamente in mezzo
        X_test = np.array([[1.5, 1.5]])
        pred = knn.predict(X_test)
        self.assertIn(pred[0], [2, 4]) # La scelta deve essere tra le due classi










if __name__ == "__main__":
    unittest.main()



