import numpy as np
import pandas as pd
import matplotlib as plt

# We need to extract:
# Clump Thickness: column 6
# Uniformity of Cell Size: column 5
# Uniformity of Cell Shape: column 11
# Marginal Adhesion: column 8
# Single Epithelial Cell Size: column 4
# Bare Nuclei: column 12
# Bland Chromatin: column 9
# Normal Nucleoli: column 3
# Mitoses: column 1

#Gestisce il caricamento del dataset
class Data_Loader():
    def __init__(self, filepath, features, classes, rename_map=None):
        self.filepath = filepath
        self.raw_data = None
        self.features_names = features
        self.classes = classes
        self.rename_map=rename_map
        self.X = None
        self.Y = None


    def load_dataset(self): #stampa del messaggio di successo/insuccesso
        try:
            self.raw_data = pd.read_csv(self.filepath)
            self.raw_data.replace(',', '.', regex = True, inplace = True)
            self.raw_data = self.raw_data.apply(pd.to_numeric, errors='coerce') #trasforma eventuali testi residui in err o NaN

            #Rinomina le colonne con pandas
            if self.rename_map is not None:
                self.raw_data.rename(columns=self.rename_map, inplace=True)
                self.features_names=[self.rename_map.get(f,f) for f in self.features_names]

            print ('Il dataset è stato caricato con successo')
            return self.raw_data
        except FileNotFoundError:
            print (f"Il dataset non è stato caricato, controlla che sia: {self.filepath}")


    def features_cleaning_and_extraction (self): #Esegue l'estrazione delle features e la sua pulizia
        if self.raw_data is None:
            print ('Impossibile pulire  dati, il dataset non è stato caricato correttamente')
            return

        # Copia del dataframe
        data_copy = self.raw_data.copy()
        print(f"Righe prima della pulizia: {len(data_copy)}")

        data_copy.dropna(subset = [self.classes], inplace = True)

        print(f"Righe dopo la pulizia: {len(data_copy)}")

        # Estrazione features e classi
        DataFrame = data_copy[self.features_names].copy()
        Classi = data_copy[self.classes].copy()

        # Fase di imputazione con media dopo pulizia delle classi
        columns_mean = DataFrame.mean()
        DataFrame.fillna(columns_mean, inplace = True)
        DataFrame.drop_duplicates(inplace = True)

        # Resetta l'indice per averli consecutivi (0, 1, 2, ... 614)
        # drop=True evita di creare una nuova colonna con i vecchi indici
        DataFrame.reset_index(drop=True, inplace=True)
        Classi.reset_index(drop=True, inplace=True)

        self.X = DataFrame
        self.Y = Classi

        self.X = DataFrame
        self.Y = Classi
        print(self.X)
        print(self.Y)

if __name__ == "__main__":
    #Nomi originali delle colonne nel file csv
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

    #Rinominazione di colonne
    rename_map= {
        'clump_thickness_ty': 'Clump Thickness',
        'uniformity_cellsize_xx': 'Uniformity of Cell Size',
        'bareNucleix_wrong': 'Bare Nuclei'
    }


    classes = "classtype_v1"
    filepath = 'Dataset_Tumori.csv'
    loader = Data_Loader(filepath, selected_features, classes, rename_map)

    loader.load_dataset()
    loader.features_cleaning_and_extraction()





















