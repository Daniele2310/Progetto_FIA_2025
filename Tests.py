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


