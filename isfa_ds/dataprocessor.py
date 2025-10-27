import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any

# =============================================================
# Data Processing Class 
# =============================================================

class DataProcessor:
    def __init__(self, molecular_data: pd.DataFrame, clinical_data: pd.DataFrame, target_data: pd.DataFrame) -> None:
        """
        Initialize the DataProcessor with molecular, clinical, and target data.
        """
        self.molecular_data : pd.DataFrame = molecular_data
        self.clinical_data : pd.DataFrame = clinical_data
        self.target_data : pd.DataFrame = target_data
        self.merged_table : pd.DataFrame = None


    def get_data_shapes(self) -> tuple:
        """
        Returns the shapes of the molecular, clinical, and target datasets.
        """
        return (self.molecular_data.shape, self.clinical_data.shape, self.target_data.shape)

    def value_types(self) -> dict:
        """
        Returns the data types of each column in the datasets.
        """
        return {
            'molecular_data_types': self.molecular_data.dtypes.to_dict(),
            'clinical_data_types': self.clinical_data.dtypes.to_dict(),
            'target_data_types': self.target_data.dtypes.to_dict()
        }
        
    def show(self) -> None:
        """
        Display the current format of the Data
        """
        if self.merged_table is None : 
            print("Molecular Data :")
            print(self.molecular_data.head())
            print("\nClinical Data :")
            print(self.clinical_data.head())    
            print("\nTarget Data :")
            print(self.target_data.head())
        else :
            print("Merged Data table : ")
            print(self.merged_table.head())
        
    def mergedatabis(self) -> pd.DataFrame:
        """
        Merges Molecular, clinical and target data on ID. 
        - Keeps multiple rows per patient. These rows are treated in different ways by another method
        """
        merged : pd.DataFrame = self.clinical_data.merge(self.molecular_data, on = "ID", how="left").merge(self.target_data, on="ID", how="left")
        self.merged_table = merged
        return merged

    
    def preprocess(self) -> tuple:
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.molecular_data.iloc[:, 1:] = imputer.fit_transform(self.molecular_data.iloc[:, 1:])
        self.clinical_data.iloc[:, 1:] = imputer.fit_transform(self.clinical_data.iloc[:, 1:])

        # Standardize features
        scaler = StandardScaler()
        self.molecular_data.iloc[:, 1:] = scaler.fit_transform(self.molecular_data.iloc[:, 1:])
        self.clinical_data.iloc[:, 1:] = scaler.fit_transform(self.clinical_data.iloc[:, 1:])

        return self.molecular_data, self.clinical_data, self.target_data
    
    def plot_quant_var(self) -> None:
        """
        Plots quantitative variables from molecular and clinical datasets.
        """
        # Plotting molecular data
        quant_cols = self.molecular_data.select_dtypes(include=[np.number]).columns.tolist()

        self.molecular_data[quant_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Molecular Data Distributions")
        plt.show()
        
        # Plotting clinical data
        quant_cols = self.clinical_data.select_dtypes(include=[np.number]).columns.tolist()

        self.clinical_data[quant_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Clinical Data Distributions")
        plt.show()

# =============================================================
# Implementation example
# =============================================================

if __name__ == "__main__":
    
    Xtrain_molecular: str = r"C:\Users\mouad\Desktop\ISFA\M2\QRT\Initial data\X_train\molecular_train.csv"
    Xtrain_clinical: str = r"C:\Users\mouad\Desktop\ISFA\M2\QRT\Initial data\X_train\clinical_train.csv"
    target_train: str = r"C:\Users\mouad\Desktop\ISFA\M2\QRT\Initial data\target_train.csv"

    data_train_mol: pd.DataFrame = pd.read_csv(Xtrain_molecular)
    data_train_cli: pd.DataFrame = pd.read_csv(Xtrain_clinical)
    target_df: pd.DataFrame = pd.read_csv(target_train)

    # Convert 'OS_YEARS' to numeric, forcing errors to NaN 

    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')

    # Ensure 'OS_STATUS' is boolean
    target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)
    print(f" Shape des données moléculaires {data_train_mol.shape} \n shape des données cliniques {data_train_cli.shape} \n shape des données cibles {target_df.shape}")

    b = DataProcessor(data_train_cli, data_train_mol, target_df)
    












