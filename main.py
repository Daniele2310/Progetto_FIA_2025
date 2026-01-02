import argparse
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os

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

class ValidationStrategy(ABC):
    """Intefaccia comune per i metodi di validazione"""
    @abstractmethod
    def validate(self, X, Y, args):
        pass

class HoldoutStrategy(ValidationStrategy):
    def validate(self, X, Y, args):
        res=holdout(X, Y, test_size=args.test_size, random_state=args.seed)
        return [res] if res else []

class RandomSubsamplingStrategy(ValidationStrategy):
    def validate(self, X, Y, args):
        return random_subsampling(X, Y, test_size=args.test_size, n=args.n_iter, random_state=args.seed)

class BootstrapStrategy(ValidationStrategy):
    def validate(self, X, Y, args):
        return bootstrap(X, Y, k=args.k_boot, random_state=args.seed)

#Implementazione della factory

class ValidationFactory:
    @staticmethod
    def get_strategy(method_name):
        method_name = method_name.lower()
        strategies={
            'holdout': HoldoutStrategy(),
            'random_subsampling': RandomSubsamplingStrategy(),
            'bootstrap': BootstrapStrategy()
        }
        aliases = {
            'holdout': ['holdout', 'hold'],
            'random_subsampling': ['random', 'subsampling', 'random_subsampling', 'rs'],
            'bootstrap': ['bootstrap', 'boot']
        }

        if method_name in strategies:
            return method_name, strategies[method_name]

        for canonical_name, names in aliases.items():
            if method_name in names:
                return canonical_name, strategies[canonical_name]

        return None,None


def parse_args():
    parser = argparse.ArgumentParser(description = "Classificazione con KNN")

    parser.add_argument("--file", type = str,  default = "Dataset_Tumori.csv" )

    parser.add_argument("--method", type = str, default="holdout",
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

def main():
    args = parse_args()

    print("--- Configurazione ---")
    print(f"File: {args.file}")
    print(f"Metodo: {args.method}")
    print(f"K-NN Neighbors: {args.k_nn}")
    print(f"Seed: {args.seed}")
    print("----------------------\n")

    classes = "classtype_v1"

    selected_features = [
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

    loader = Data_Loader(args.file, selected_features, classes)
    df = loader.load_dataset()
    print(df)

    if df is None:
        print("Errore: Impossibile caricare il dataset")
        return

    columns = df.columns.tolist()
    if classes in columns:
        columns.remove(classes)

    features = [col for col in columns if col in selected_features]

    loader.features_names = features

    loader.features_cleaning_and_extraction()
    X = loader.X
    Y = loader.Y

    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc[:,0]

    splits = []

    #Ottiene la strategia tramite Factory
    method_name, strategy=ValidationFactory.get_strategy(args.method)

    if strategy is None:
        print (f"Errore: Metodo di validazione {args.method} non supportato.")
        return
    splits= strategy.validate(X, Y, args)

    knn = KNN_Classifier(K = args.k_nn)
    all_metrics = []
    for i, (X_train, X_test, Y_train, Y_test) in enumerate(splits):
        knn.fit(X_train, Y_train)
        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)

        evaluator = MetricsEvaluator(Y_test, y_pred, Y_scores=y_proba, pos_label=4)
        metrics = evaluator.get_metrics()
        all_metrics.append(metrics) #salva le metriche ad ogni split
        print(metrics)
        evaluator.plot_confusion_matrix()

    if all_metrics:
        print("\n--- Media Prestazioni su tutti gli esperimenti ---")
        df_metrics=pd.DataFrame(all_metrics)
        print(df_metrics.mean())



    #i risultati del validation vengono riportati in un file excel all'interno di una cartella
    if all_metrics:
        results_dir = "results"
        method_dir = os.path.join(results_dir, method_name)

        # crea cartelle se non esistono
        os.makedirs(method_dir, exist_ok=True)

        df_metrics = pd.DataFrame(all_metrics)

        excel_path = os.path.join(
            method_dir,
            f"metrics_{args.method}.xlsx"
        )

        df_metrics.to_excel(excel_path, index=False)

        print(f"\nMetriche salvate in: {excel_path}")



if __name__ == "__main__":
    main()