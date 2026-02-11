# Progetto_FIA_2025
Progetto di Machine Learning per la classificazione di tumori come benigni (classe 2) o maligni (classe 4) utilizzando un classificatore k-NN.

## Descrizione
Il progetto implementa una pipeline completa di machine learning per classificare cellule tumorali basandosi su 9 features:
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses


## Requisiti

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Installazione

1. Clonare il repository:
```bash
git clone https://github.com/Daniele2310/Progetto_FIA_2025
cd <NOME_CARTELLA>
```

2. Creare un virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # Linux/MacOS
# oppure
env\Scripts\activate     # Windows
```

3. Installare le dipendenze:
```bash
pip install -r requirements.txt
```


## Opzioni di Input
L'utente può specificare i seguenti parametri:

### 1. Numero di vicini (k)
Numero di vicini da considerare per il classificatore k-NN.

### 2. Metodo di validazione
- **Holdout**: divisione singola in training e test set
  - Parametro: percentuale del test set
- **Random Subsampling**: con K numero di holdout
- **Bootstrap**

### 3. Metriche di valutazione
- Accuracy Rate
- Error Rate
- Sensitivity
- Specificity
- Geometric Mean
- ROC
- Area Under the Curve (AUC)

## Come eseguire il codice
Il programma viene gestito tramite lo script principale `main.py`. È possibile configurare l'esecuzione utilizzando il menù interattivo per scegliere i metodi di validazione e gli iperparametri della classificazione:

* **Comando base**:
    ```bash
    python main.py
    ```


## Visualizzazione e Interpretazione dei Risultati
Al termine dell'elaborazione, il sistema fornisce diversi strumenti per valutare l'efficacia del modello:

### 1. Risultati e Report
- **Output a video**: Vengono mostrate tutte le metriche calcolate (Accuracy, Sensitivity, ecc.). Se si utilizzano metodi iterativi (Subsampling o Bootstrap), il sistema calcola e mostra la **media delle prestazioni** attraverso tutti i K esperimenti effettuati.
- **File Excel**: I risultati dettagliati di ogni esperimento vengono salvati automaticamente in un file `.xlsx` nella cartella `results/` per consentire analisi comparative

### 2. Interpretazione Grafica
- **Confusion Matrix**: mostra la relazione tra le predizioni del modello e le classi reali
- **Curva ROC e AUC**: visualizza il trade-off tra Sensibilità e Specificità. Il valore **AUC (Area Under the Curve)** fornisce un indice sintetico della capacità discriminante: più il valore è vicino a 1.0, migliore è la capacità del classificatore di distinguere correttamente tra tumori benigni e maligni.


## Autori (gruppo n.6)

- Nicole Bovolenta
- Daniele Cantagallo
- Luca Tortoriello
