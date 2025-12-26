import argparse
import pandas as pd
import numpy as np

from Data_Preprocessing import Data_Loader
from Data_Preprocessing import FileConverter
from holdout import holdout
from random_subsampling import random_subsampling
from Bootstrap import bootstrap
from Model_Development import KNN_Classifier
from MetricsEvaluation import MetricsEvaluator



'''
1. file converter
2. data loader -> dataset pulito come file.csv
3. KNN
4. validation (su richiesta holdout, random subsampling o bootstrap)
5. Per ogni validation, metriche (scelte dall'utente) salvate su file Excel e plot della CM e ROC+AUC
'''



def parse_args():
    parser = argparse.ArgumentParser(description = "Classificazione con KNN")

    parser.add_argument("--file", type = str,  default = "Dataset_Tumori.csv" )

    parser.add_argument("--method", type = str, choices = ['holdout', 'subsampling', 'bootstrap'], default="holdout",
                        help = "Metodo di validazione: 'holdout', 'subsampling', 'bootstrap'")

    parser.add_argument("--test_size", type = float, default = 0.2, help = "Percentuale del test set usato per holdout e subsampling")

    parser.add_argument('--n_iter', type = int, default = 10,
                        help = "Numero di ripetizioni per Random Subsampling (default: 10)")

    parser.add_argument('--k_nn', type=int, default = 8,
                        help="Valore di k per l'algoritmo KNN (default: 8)")

    parser.add_argument('--k_boot', type=int, default = 10,
                        help="Numero di campionamenti per Bootstrap (default: 10)")

    parser.add_argument('--seed', type=int, default=42,
                        help="Seed casuale per la riproducibilità")

    return parser.parse_args()