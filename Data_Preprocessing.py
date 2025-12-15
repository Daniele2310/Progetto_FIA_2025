import numpy as np
import pandas as pd
import matplotlib as plt
import json


class FileConverter:
    """Gestisce la conversione di file con formato diverso da .csv"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def convert_to_csv(self):
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

#Gestione di caricamento del dataset
class DataLoader:

    def __init__(self, filepath, features, classes, rename_map=None):
        # Converte in CSV se necessario
        converter = FileConverter(filepath)
        self.filepath = converter.convert_to_csv()

        self.raw_data = None
        self.features_names = features
        self.classes = classes
        self.rename_map = rename_map
        self.X = None
        self.Y = None

    def load_dataset(self):
        try:
            self.raw_data = pd.read_csv(self.filepath)
            self.raw_data.replace(',', '.', regex=True, inplace=True)
            self.raw_data = self.raw_data.apply(pd.to_numeric, errors='coerce')

            if self.rename_map is not None:
                self.raw_data.rename(columns=self.rename_map, inplace=True)
                self.features_names = [self.rename_map.get(f, f) for f in self.features_names]

            print('Il dataset è stato caricato con successo')
            return self.raw_data
        except FileNotFoundError:
            print(f"Il dataset non è stato caricato, controlla che sia: {self.filepath}")

    def features_cleaning_and_extraction(self):
        if self.raw_data is None:
            print('Impossibile pulire dati, il dataset non è stato caricato correttamente')
            return

        data_copy = self.raw_data.copy()
        print(f"Righe prima della pulizia: {len(data_copy)}")

        data_copy.dropna(subset=[self.classes], inplace=True)
        print(f"Righe dopo la pulizia: {len(data_copy)}")

        DataFrame = data_copy[self.features_names].copy()
        Classi = data_copy[self.classes].copy()

        columns_mean = DataFrame.mean()
        DataFrame.fillna(columns_mean, inplace=True)
        DataFrame.drop_duplicates(inplace=True)

        DataFrame.reset_index(drop=True, inplace=True)
        Classi.reset_index(drop=True, inplace=True)

        self.X = DataFrame
        self.Y = Classi

        print(self.X)
        print(self.Y)


if __name__ == "__main__":
    features = ['Blood Pressure',
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

    rename_map = {
        'clump_thickness_ty': 'Clump Thickness',
        'uniformity_cellsize_xx': 'Uniformity of Cell Size',
        'bareNucleix_wrong': 'Bare Nuclei'
    }

    classes = "classtype_v1"
    filepath = 'Dataset_Tumori.csv'  # Può essere anche .xlsx, .json, .txt, .tsv

    loader = DataLoader(filepath, selected_features, classes, rename_map)
    loader.load_dataset()
    loader.features_cleaning_and_extraction()