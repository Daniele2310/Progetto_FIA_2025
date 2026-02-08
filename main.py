import argparse
from unittest import case

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os
from datetime import datetime

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
5. Per ogni validation, metriche salvate su file Excel + plot CM e ROC
'''

# ===================== VALIDATION STRATEGY =====================

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
        # method_name = method_name.lower()
        match method_name:
            case "holdout":
                return HoldoutStrategy()
            case "random_subsampling":
                return RandomSubsamplingStrategy()
            case "bootstrap":
                return BootstrapStrategy()

        # strategy = method_name
        """
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
         """


# ===================== ARGPARSE =====================

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


# ===================== MAIN =====================

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

    if df is None:
        print("Errore: Impossibile caricare il dataset")
        return

    # selezione feature realmente presenti
    columns = df.columns.tolist()
    if classes in columns:
        columns.remove(classes)

    features = [f for f in selected_features if f in columns]
    loader.features_names = features

    loader.features_cleaning_and_extraction()
    X = loader.X
    Y = loader.Y

    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc[:, 0]

    splits = []
    # Ottiene la strategia tramite Factory
    # method_name, strategy=ValidationFactory.get_strategy(args.method)
    while True:
        print("MENU PRINCIPALE")
        print("1. Esegui Holdout")
        print("2. Esegui Random Subsampling")
        print("3. Esegui Bootstrap")
        print("4. Non eseguire nulla")

        try:
            scelta = int(input("Inserisci la scelta (1-4): "))
        except ValueError:
            print("Inserire una scelta valida")
            continue

        if scelta == 1:
            method_name = "holdout"
            strategy = ValidationFactory.get_strategy(method_name)
            splits = strategy.validate(X, Y, args)
        elif scelta == 2:
            method_name = "random_subsampling"
            strategy = ValidationFactory.get_strategy(method_name)
            splits = strategy.validate(X, Y, args)
        elif scelta == 3:
            method_name = "bootstrap"
            strategy = ValidationFactory.get_strategy(method_name)
            splits = strategy.validate(X, Y, args)
        elif scelta == 4:
            print("Non sto eseguendo nulla")
            break
        break
    """
    if strategy is None:
        print (f"Errore: Metodo di validazione {args.method} non supportato.")
        return
    """

    # splits = strategy.validate(X, Y, args)
    knn = KNN_Classifier(K = args.k_nn)
    all_metrics = []

    for i, (X_train, X_test, Y_train, Y_test) in enumerate(splits):
        knn.fit(X_train, Y_train)
        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)

        evaluator = MetricsEvaluator(
            Y_test, y_pred,
            Y_scores=y_proba,
            pos_label=4
        )

        metrics = evaluator.get_metrics()

        # === TIMESTAMP PER OGNI ESPERIMENTO ===
        metrics_with_time = metrics.copy()
        metrics_with_time["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        all_metrics.append(metrics_with_time)

        print(metrics)
        evaluator.plot_confusion_matrix()
        evaluator.plot_roc_curve()

    # ===================== EXCEL =====================

    if all_metrics:
        results_dir = "results"
        method_dir = os.path.join(results_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        df_metrics = pd.DataFrame(all_metrics).round(2)

        # numero esperimento
        df_metrics.insert(0, "Esperimento", range(1, len(df_metrics) + 1))

        # ordine colonne: Esperimento, Timestamp, metriche
        cols = ["Esperimento", "Timestamp"] + [
            c for c in df_metrics.columns if c not in ["Esperimento", "Timestamp"]
        ]
        df_metrics = df_metrics[cols]

        # riassunto
        summary = pd.DataFrame({
            "Media": df_metrics.drop(columns=["Esperimento", "Timestamp"]).mean(),
            "Deviazione Std": df_metrics.drop(columns=["Esperimento", "Timestamp"]).std()
        }).round(2)

        excel_path = os.path.join(
            method_dir,
            f"metrics_{args.method}.xlsx"
        )

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_metrics.to_excel(writer, sheet_name="Risultati", index=False)
            summary.to_excel(writer, sheet_name="Riassunto")

            workbook = writer.book
            ws_r = writer.sheets["Risultati"]
            ws_s = writer.sheets["Riassunto"]

            header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'middle',
                'border': 1,
                'bg_color': '#D9E1F2'
            })

            cell_format = workbook.add_format({
                'align': 'center',
                'border': 1
            })

            for col, name in enumerate(df_metrics.columns):
                ws_r.write(0, col, name, header_format)
                ws_r.set_column(col, col, 18, cell_format)

            ws_s.write(0, 0, "", header_format)
            ws_s.write(0, 1, "Media", header_format)
            ws_s.write(0, 2, "Deviazione Std", header_format)
            ws_s.set_column(0, 0, 22, cell_format)
            ws_s.set_column(1, 2, 18, cell_format)

        print(f"\nMetriche salvate in: {excel_path}")


if __name__ == "__main__":
    main()

