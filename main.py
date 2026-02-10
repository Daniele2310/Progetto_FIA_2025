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
        print(f"\n[INFO] Esecuzione Holdout (Test: {args.test_size*100}%, Seed: {args.seed})...")
        res = holdout(X, Y, test_size=args.test_size, random_state=args.seed)
        return [res] if res else []

class RandomSubsamplingStrategy(ValidationStrategy):
    def validate(self, X, Y, args):
        print(f"\n[INFO] Esecuzione Random Subsampling ({args.n_iter} iterazioni)...")
        return random_subsampling(X, Y, test_size=args.test_size, n=args.n_iter, random_state=args.seed)

class BootstrapStrategy(ValidationStrategy):
    def validate(self, X, Y, args):
        print(f"\n[INFO] Esecuzione Bootstrap ({args.k_boot} campionamenti)...")
        return bootstrap(X, Y, k=args.k_boot, random_state=args.seed)

# Implementazione della factory

class ValidationFactory:
    @staticmethod
    def get_strategy(method_name):
        match method_name:
            case "holdout":
                return HoldoutStrategy()
            case "random_subsampling":
                return RandomSubsamplingStrategy()
            case "bootstrap":
                return BootstrapStrategy()
        return None

# ===================== HELPER INPUT =====================

def get_user_float(prompt, min_val=0.0, max_val=1.0, default=0.2):
    """Richiede un input float sicuro all'utente"""
    while True:
        try:
            inp = input(f"{prompt} [Default: {default}]: ").strip()
            if not inp:
                return default
            val = float(inp)
            if min_val < val < max_val:
                return val
            else:
                print(f"Inserire un valore tra {min_val} e {max_val}.")
        except ValueError:
            print("Input non valido. Inserire un numero con la virgola (es. 0.3).")

def get_user_int(prompt, min_val=1, default=5):
    """Richiede un input int sicuro all'utente"""
    while True:
        try:
            inp = input(f"{prompt} [Default: {default}]: ").strip()
            if not inp:
                return default
            val = int(inp)
            if val >= min_val:
                return val
            else:
                print(f"Inserire un valore maggiore o uguale a {min_val}.")
        except ValueError:
            print("Input non valido. Inserire un numero intero.")


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
    # Nota: I valori stampati qui sono i default iniziali, verranno sovrascritti dal menu
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

    splits = [] # Inizializzazione lista risultati
    method_name = "" # Per tenere traccia del metodo scelto

    # Ottiene la strategia tramite Factory
    while True:
        print("\n" + "="*30)
        print(" MENU PRINCIPALE")
        print("="*30)
        print("1. Esegui Holdout")
        print("2. Esegui Random Subsampling")
        print("3. Esegui Bootstrap")
        print("4. Esci")

        try:
            inp = input("Inserisci la scelta (1-4): ").strip()
            if not inp: continue
            scelta = int(inp)
        except ValueError:
            print("Inserire una scelta valida")
            continue

        if scelta == 4:
            print("Uscita dal programma.")
            return

        # Configurazione Parametri in base alla scelta
        if scelta == 1:
            method_name = "holdout"
            # Scelta test set size
            args.test_size = get_user_float("Inserisci la dimensione del Test Set (es. 0.3 per 30%)", default=0.3)
            # Scelta K del KNN
            args.k_nn = get_user_int("Inserisci il valore di K per il K-NN", default=5)
            
        elif scelta == 2:
            method_name = "random_subsampling"
            # Scelta test set size
            args.test_size = get_user_float("Inserisci la dimensione del Test Set (es. 0.3 per 30%)", default=0.3)
            # Numero iterazioni
            args.n_iter = get_user_int("Inserisci il numero di iterazioni (split)", default=10)
            # Scelta K del KNN
            args.k_nn = get_user_int("Inserisci il valore di K per il K-NN", default=5)

        elif scelta == 3:
            method_name = "bootstrap"
            # Numero campionamenti bootstrap
            args.k_boot = get_user_int("Inserisci il numero di campionamenti Bootstrap", default=10)
            # Scelta K del KNN
            args.k_nn = get_user_int("Inserisci il valore di K per il K-NN", default=5)
        
        else:
            print("Scelta non valida.")
            continue

        # Aggiorna il metodo negli args per coerenza (usato poi nel nome file Excel)
        args.method = method_name
        
        # Recupera ed esegue la strategia scelta
        strategy = ValidationFactory.get_strategy(method_name)
        splits = strategy.validate(X, Y, args)

        # Se splits è vuoto o None, significa che qualcosa è andato storto nella validazione (es. dataset troppo piccolo)
        if not splits:
            print("ATTENZIONE: Nessuno split generato. Torno al menu.")
            continue
            
        break # Esce dal while solo se la strategia è stata eseguita con successo

    # === FASE DI TRAINING E VALIDAZIONE ===
    
    knn = KNN_Classifier(K = args.k_nn)
    all_metrics = []
    
    print(f"\nAvvio Training e Valutazione su {len(splits)} iterazione/i...")

    for i, (X_train, X_test, Y_train, Y_test) in enumerate(splits):
        iterazione_msg = f"ITERAZIONE {i+1} / {len(splits)}"
        print("\n" + "-"*len(iterazione_msg))
        print(iterazione_msg)
        print("-"*(len(iterazione_msg)))

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

        
        # === STAMPA FORMATTATA ===
        print("\n" + "┌" + "─"*35 + "┐")
        print(f"│ {'METRICA':<22} │ {'VALORE':>8} │")
        print("├" + "─"*35 + "┤")
        
        for k, v in metrics.items():
            print(f"│ {k:<22} │ {v:>8} │")
            
        print("└" + "─"*35 + "┘")



        input(f"\n>>> Premi [INVIO] per visualizzare i grafici dell'iterazione {i+1}...")
        
        # Nota: I plot potrebbero bloccare l'esecuzione finché non vengono chiusi, dipende dal backend matplotlib
        print("Visualizzazione grafici in corso... (Chiudi la finestra del grafico per continuare)")
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

        # Try-Except per gestire la mancanza di xlsxwriter
        try:
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

            print(f"\n[SUCCESSO] Metriche salvate in: {excel_path}")
            
        except ModuleNotFoundError:
            print("\n[ATTENZIONE] Modulo 'xlsxwriter' non trovato. Salvataggio in formato CSV standard.")
            csv_path = excel_path.replace('.xlsx', '.csv')
            df_metrics.to_csv(csv_path, index=False)
            print(f"Metriche salvate in: {csv_path}")

        except Exception as e:
            print(f"\n[ERRORE] Impossibile salvare i risultati su file: {e}")

if __name__ == "__main__":
    main()
