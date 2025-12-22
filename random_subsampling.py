from holdout import holdout
from Data_Preprocessing import Data_Loader

def random_subsampling(X, Y, test_size=0.2, n=1, random_state=42):
    # n = numero di ripetizioni
    """
     Due casi:
    - holdout se n == 1
    - random subsampling se n > 1
    """

    # holdout
    if n == 1:
        return holdout(X, Y, test_size=test_size, random_state=random_state)

    results = [] #per memorizzare i risultati degli holdout

    for i in range(n):
        # in questo caso ci interessa che ogni volta che il codice viene
        # eseguito, il seed cambi, altrimenti la suddivisione sarebbe la stessa
        #
        current_seed = random_state + i  # modifica il seed ad ogni iterazione del ciclo for

        split = holdout(
            X,
            Y,
            test_size=test_size,
            random_state=current_seed
        )

            # split è una tupla con questa forma -> split = (X_train, X_test, Y_train, Y_test)

        if split is not None:
            results.append(split)

    return results
