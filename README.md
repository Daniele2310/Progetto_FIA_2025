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



### Output
- File Excel con le performance
- Grafici (Confusion Matrix, ROC Curve, AUC)


## Autori (gruppo n.6)

- Nicole Bovolenta
- Daniele Cantagallo
- Luca Tortoriello
