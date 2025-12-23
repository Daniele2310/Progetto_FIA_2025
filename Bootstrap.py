import numpy as np

def bootstrap(X, Y, k, random_state):
    X = np.array(X)
    Y = np.array(Y)
    n_samples = X.shape[0]

    dataset_split = []
    for i in range(k):
        current_seed = random_state + i
        np.random.seed(current_seed)
        train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_train = X[train_indices]
        Y_train = Y[train_indices]

        test_indices = list(set(range(n_samples)) - set(train_indices))
        X_test = X[test_indices]
        Y_test = Y[test_indices]

        dataset_split.append([X_train, Y_train, X_test, Y_test])

    if len(dataset_split[0][2]) / n_samples < 0.30:
        print("ATTENZIONE: stai utilizzando un test test_set troppo basso")
        risposta = input("Desideri continuare? (s/n): ").strip().lower()

        if risposta != 's':
            print("Esecuzione interrotta dall'utente.")
            return None

    return dataset_split




