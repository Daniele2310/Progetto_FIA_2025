import numpy as np
import pandas as pd
import json
import plotly.express as px
from typing import List, Dict, Optional, Any, Union



class FileConverter:
    """Gestisce la conversione di file con formato diverso da .csv"""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.data = None

    def convert_to_csv(self) -> str:
        """Converte il file in CSV se necessario"""

        if self.filepath.endswith('.csv'):
            return self.filepath

        if self.filepath.endswith('.xlsx'):
            self.data = pd.read_excel(self.filepath)

        elif self.filepath.endswith('.tsv'):
            self.data = pd.read_csv(self.filepath, delimiter='\t')

        elif self.filepath.endswith('.txt'):
            for delim in ['\t', ';', ',', '|', ' ']:
                try:
                    self.data = pd.read_csv(self.filepath, delimiter=delim)
                    if len(self.data.columns) > 1:
                        break
                except:
                    continue

        elif self.filepath.endswith('.json'):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            if isinstance(json_data, list):
                self.data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                for key, value in json_data.items():
                    if isinstance(value, list):
                        self.data = pd.DataFrame(value)
                        break
                else:
                    self.data = pd.DataFrame([json_data])
        else:
            raise ValueError(f"Formato non supportato: {self.filepath}")

        csv_path = self.filepath.rsplit('.', 1)[0] + '.csv'
        self.data.to_csv(csv_path, index=False)
        print(f"File convertito in: {csv_path}")

        return csv_path

#Gestisce il caricamento del dataset
class Data_Loader():
    def __init__(self,
                 filepath: str,
                 features: List[str],
                 classes: str,
                 rename_map=None) -> None:
        # Converte in CSV se necessario
        converter = FileConverter(filepath)
        self.filepath = converter.convert_to_csv()

        self.raw_data = None
        self.features_names = features
        self.classes = classes
        self.rename_map = rename_map
        self.X = None
        self.Y = None

    def load_dataset(self) -> Optional[pd.DataFrame]: #stampa del messaggio di successo/insuccesso
        try:
            self.raw_data = pd.read_csv(self.filepath)
            self.raw_data.replace(',', '.', regex=True, inplace=True)
            self.raw_data = self.raw_data.apply(pd.to_numeric, errors='coerce')

            if self.rename_map is not None:
                self.raw_data.rename(columns=self.rename_map, inplace=True)
                self.features_names = [self.rename_map.get(f, f) for f in self.features_names]

            print("Il dataset è stato caricato con successo")
            return self.raw_data
        except FileNotFoundError:
            print(f"Il dataset non è stato caricato, controlla che sia: {self.filepath}")

    def features_cleaning_and_extraction(self) -> None:
        if self.raw_data is None:
            print(" Impossibile pulire dati, il dataset non è stato caricato correttamente")
            return

        print("\n" + "═" * 62)
        print(" INIZIO PROCESSO DI PULIZIA DATI")
        print("═" * 62 + "\n")

        data_copy = self.raw_data.copy()
        initial_rows = len(data_copy)
        print(f"\n RIGHE INIZIALI DEL DATASET: {initial_rows}")
        print("-" * 62)

        # Rimozione colonna ID (Sample Code Number)
        column_id = 'Sample code number'
        if column_id in data_copy.columns:
            data_copy.drop(columns = [column_id], inplace=True)
            print(f" Colonna '{column_id}' eliminata")
            print(f" (Operazione necessaria per identificare duplicati reali)")
        else:
            print(f" Colonna '{column_id}' non trovata nel dataset (o già rimossa)")
        print("-" * 62)

        # Controllo Valori Fuori Range (<0 oppure >10)
        print("\n ANALISI VALORI FUORI RANGE [0 - 10]")
        out_of_range_found = False
        current_features = [f for f in self.features_names if f in data_copy.columns]
        for col in current_features:
            mask_out = (data_copy[col] > 10) | (data_copy[col] < 0)
            count_out = mask_out.sum()

            if count_out > 0:
                out_of_range_found = True
                # Calcolo quanti positivi e negativi per dettaglio
                over_10 = (data_copy[col] > 10).sum()
                under_0 = (data_copy[col] < 0).sum()
                print(f" Colonna '{col}': {over_10} valori > 10 | {under_0} valori < 0 (verranno settati a NaN)")
                data_copy.loc[mask_out, col] = np.nan
        if not out_of_range_found:
            print(" Nessun valore fuori range trovato nelle features selezionate.")

        # Pulizia righe con troppi Nan
        print("\n RIMOZIONE RIGHE NON VALIDE")
        nans_per_raw = data_copy.isnull().sum(axis=1) # axis=1 poichè cosi faccio il check sulle righe
        rows_before_nan_drop = len(data_copy)
        data_copy = data_copy[nans_per_raw <= 5]
        rows_dropped_nan = rows_before_nan_drop - len(data_copy)
        print(f" Righe eliminate per troppi dati mancanti (>5 NaN): ... - {rows_dropped_nan}")

        # Pulizia Classi Mancanti
        if self.classes in data_copy.columns:
            rows_before_class_drop = len(data_copy)
            data_copy.dropna(subset=[self.classes], inplace=True)
            rows_dropped_class = rows_before_class_drop - len(data_copy)
            print(f" Righe eliminate perché senza classe ({self.classes}): ... - {rows_dropped_class}")
        else:
            print(f" Colonna classe '{self.classes}' non trovata")

        # Rimozione Duplicati
        rows_before_dup_drop = len(data_copy)
        data_copy.drop_duplicates(inplace=True)
        rows_dropped_dup = rows_before_dup_drop - len(data_copy)
        print(f" Righe duplicate identiche eliminate: ... - {rows_dropped_dup}")

        # --- Riepilogo Finale ---
        print("\n" + "═" * 62)
        print(f" RIGHE FINALI DISPONIBILI: {len(data_copy)}")
        print(f" TOTALE RIGHE SCARTATE:    {initial_rows - len(data_copy)}")
        print("═" * 62 + "\n")

        available_features = [f for f in self.features_names if f in data_copy.columns]

        DataFrame = data_copy[available_features].copy()
        Classi = data_copy[[self.classes]].copy() # qui serve la doppia quadra perché così sto creando un dataframe con una sola series

        # Fase di imputazione con mediana dopo pulizia delle classi
        columns_median = DataFrame.median()
        DataFrame.fillna(columns_median, inplace = True)

        # Resetta l'indice per averli consecutivi (0, 1, 2, ... 614)
        # drop=True evita di creare una nuova colonna con i vecchi indici
        DataFrame.reset_index(drop=True, inplace=True)
        Classi.reset_index(drop=True, inplace=True)

        self.X = DataFrame
        self.Y = Classi

        print(" PULIZIA COMPLETATA CON SUCCESSO.")
        input(" Premi [INVIO] per proseguire... ")

# Da inserire nel main.py
if __name__ == "__main__":
    features= ['Blood Pressure',
               'Mitoses',
               'Sample code number',
               'Normal Nucleoli',
               'Single Epithelial Cell Size',
               'uniformity_cellsize_xx',
               'clump_thickness_ty',
               'Heart Rate',
               'Marginal Adhesion',
               'Bland Chromatin',
               'Uniformity of Cell Shape',
               'bareNucleix_wrong']
    selected_features = [features[i] for i in [6, 5, 10, 8, 4, 11, 9, 3, 1]]

    classes = "classtype_v1"
    filepath = 'Dataset_Tumori.csv'
    loader = Data_Loader(filepath, selected_features, classes)

    loader.load_dataset()
    loader.features_cleaning_and_extraction()

    fig = px.pie(loader.Y, classes, color_discrete_sequence=['#ffffd4 ', '#fe9929 '],
                 title='Data Distribution', template='plotly_dark')

    fig.show()
    # Da qui si evince come il nostro dataset non sia bilanciato: 66% appartiene alla classe 2 e il 33% alla classe 4