"""
Model Training Module for Titanic Survival Prediction
Implements Random Forest classifier with evaluation utilities.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np

class TitanicModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize Random Forest classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        self.accuracy = None
        self.feature_importance = None
        self.training_complete = False
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        """
        self.model.fit(X_train, y_train)
        self.training_complete = True
        
        if hasattr(X_train, 'columns'):
            self._calculate_feature_importance(X_train.columns)
        else:
            # Create generic feature names if no columns available
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
            self._calculate_feature_importance(feature_names)
    
    def predict(self, X):
        """
        Make predictions using trained model.
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        
        Returns:
        --------
        array: Predicted labels
        """
        if not self.training_complete:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions using trained model.
        
        Parameters:
        -----------
        X : array-like
            Features for prediction
        
        Returns:
        --------
        array: Predicted probabilities
        """
        if not self.training_complete:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            True test labels
        
        Returns:
        --------
        dict: Evaluation metrics
        """
        if not self.training_complete:
            raise ValueError("Model must be trained before evaluation.")
        
        y_pred = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print("=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        print(f"\nAccuracy: {self.accuracy:.4f}")
        print(f"Correct Predictions: {np.sum(y_pred == y_test)}/{len(y_test)}")
        
        print("\nClassification Report:")
        print("-" * 30)
        print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))
        
        print("\nConfusion Matrix:")
        print("-" * 30)
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        return {
            'accuracy': self.accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def _calculate_feature_importance(self, feature_names):
        """Calculate and store feature importance."""
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    
    def display_feature_importance(self, top_n=10):
        """Display top N feature importances."""
        if self.feature_importance is None:
            print("Feature importance not available. Train the model first.")
            return
        
        print("\n" + "=" * 50)
        print(f"TOP {top_n} FEATURE IMPORTANCES")
        print("=" * 50)
        
        top_features = self.feature_importance.head(top_n)
        for idx, row in top_features.iterrows():
            print(f"{row['Feature']:15} : {row['Importance']:.4f}")
    
    def save_model(self, path='models/random_forest_model.pkl'):
        """Save trained model to file."""
        if not self.training_complete:
            raise ValueError("Cannot save untrained model.")
        
        joblib.dump(self.model, path)
        print(f" Model saved to {path}")
    
    @staticmethod
    def load_model(path='models/random_forest_model.pkl'):
        """Load trained model from file."""
        model = joblib.load(path)
        print(f" Model loaded from {path}")
        return model