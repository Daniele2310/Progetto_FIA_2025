import numpy as np

def holdout(X, y, test_size=0.2, random_state=42):

    # Si fissa il seed del generatore casuale, così da poter garantire che ogni volta che il codice runna
    # la divisione casuale sia sempre la stessa
    np.random.seed(random_state)

    n_samples = len(X)

    # ad ogni riga del dataset si fa corrispondere un indice  [0 - (n_samples-1)]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # mescolo gli indici in modo casuale

    # (1 - test_size)% del dataset va nel training set
    split_point = int(n_samples * (1 - test_size))

    train_idx = indices[:split_point]   # primi indici → training set
    test_idx  = indices[split_point:]   # restanti indici → test set

    # selezione tramite indici
    X_train = X.iloc[train_idx]   #selezione delle features per il training set
    X_test  = X.iloc[test_idx]    #selezione delle features per il test set

    y_train = y.iloc[train_idx]   #selezione delle classi per il training set
    y_test  = y.iloc[test_idx]    #selezione delle classi per il test set



    return X_train, X_test, y_train, y_test    # Restituzione dei risultati

