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

def main():
    args = parse_args()

    print("--- Configurazione ---")
    print(f"File: {args.file}")
    print(f"Metodo: {args.method}")
    print(f"K-NN Neighbors: {args.k_nn}")
    print(f"Seed: {args.seed}")
    print("----------------------\n")

    classes = "classtype_v1"
    loader = Data_Loader(args.file, [], classes)
    df = loader.load_dataset()
    print(df)

    if df is None:
        print("Errore: Impossibile caricare il dataset")

    columns = df.columns.tolist()
    if classes in columns:
        columns.remove(classes)

    loader.features_names = columns

    loader.features_cleaning_and_extraction()
    X = loader.X
    Y = loader.Y

    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc[:,0]

    splits = []

    if args.method == 'holdout':
        res = holdout(X, Y, test_size=args.test_size, random_state=args.seed)
        if res:
            splits.append(res)

    elif args.method == 'subsampling':
        splits = random_subsampling(X, Y, test_size=args.test_size, n=args.n_iter, random_state=args.seed)

    elif args.method == 'bootstrap':
        splits = bootstrap(X, Y, k=args.k_boot, random_state=args.seed)

    if not splits:
        print("Errore nella generazione degli split o interruzione utente.")

    knn = KNN_Classifier(K = args.k_nn)
    for i, (X_train, X_test, Y_train, Y_test) in enumerate(splits):
        knn.fit(X_train, Y_train)
        y_pred = knn.predict(X_test)
        y_proba = knn.predict_proba(X_test)

        evaluator = MetricsEvaluator(Y_test, y_pred, Y_scores=y_proba, pos_label=4)
        metrics = evaluator.get_metrics()
        print(metrics)
        evaluator.plot_confusion_matrix()





if __name__ == "__main__":
    main()