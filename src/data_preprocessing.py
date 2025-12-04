"""
Data Preprocessing Module for Titanic Dataset
Handles data cleaning, feature engineering, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self, train_df=None):
        """
        Initialize preprocessor with optional training data for fitting.
        
        Parameters:
        -----------
        train_df : pd.DataFrame, optional
            Training data to calculate imputation values (prevents data leakage)
        """
        self.scaler = StandardScaler()
        self.age_median = None
        self.fare_median = None
        self.feature_names = None
        
        if train_df is not None:
            self._fit_params(train_df)
    
    def _fit_params(self, train_df):
        """Calculate parameters from training data to prevent data leakage."""
        self.age_median = train_df['Age'].median()
        self.fare_median = train_df['Fare'].median()
    
    def prepare_features(self, df, is_training=True):
        """
        Clean and engineer features from raw Titanic dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw Titanic dataset
        is_training : bool
            Whether this is training data (affects missing value handling)
        
        Returns:
        --------
        pd.DataFrame: Processed features
        """
        df = df.copy()
        
        # Map categorical variables
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        
        # Handle missing values
        age_impute = self.age_median if self.age_median else df['Age'].median()
        df['Age'].fillna(age_impute, inplace=True)
        
        fare_impute = self.fare_median if self.fare_median else df['Fare'].median()
        df['Fare'].fillna(fare_impute, inplace=True)
        
        df['Embarked'].fillna(2, inplace=True)  # 'S' is most common
        
        # Feature engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Select final features
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Embarked', 'FamilySize', 'IsAlone']
        
        self.feature_names = features
        return df[features]
    
    def scale_features(self, X, fit=False):
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X : array-like
            Features to scale
        fit : bool
            Whether to fit the scaler (True for training, False for test)
        
        Returns:
        --------
        array: Scaled features
        """
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def save_scaler(self, path='models/scaler.pkl'):
        """Save fitted scaler for later use."""
        joblib.dump(self.scaler, path)
        print(f" Scaler saved to {path}")
    
    def load_scaler(self, path='models/scaler.pkl'):
        """Load previously fitted scaler."""
        self.scaler = joblib.load(path)
        print(f" Scaler loaded from {path}")
        return self.scaler