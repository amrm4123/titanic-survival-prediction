#!/usr/bin/env python3
"""
Titanic Survival Prediction - Main Pipeline
Author: Amr AL-Kayal
Date: November 2025

Complete machine learning pipeline for predicting Titanic passenger survival.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import DataPreprocessor, TitanicModel
from src import load_data, explore_data, plot_survival_distribution, save_submission
from sklearn.model_selection import train_test_split

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'submissions', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep file if directory is empty
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'a').close()

def run_pipeline():
    """Main execution pipeline for Titanic survival prediction."""
    
    print("\n" + "="*60)
    print(" TITANIC SURVIVAL PREDICTION PIPELINE")
    print("="*60)
    print("Author: Amr AL-Kayal")
    print("Date: November 2025")
    print("="*60)
    
    # Step 1: Create directories
    print("\n STEP 1: Setting up directories...")
    create_directories()
    
    # Step 2: Load Data
    print("\n STEP 2: Loading data...")
    try:
        train_df, test_df = load_data('data/train.csv', 'data/test.csv')
    except FileNotFoundError:
        print("\n Data files not found.")
        print("Please download the Titanic dataset from Kaggle:")
        print("https://www.kaggle.com/c/titanic/data")
        print("\nRequired files:")
        print("  - data/train.csv")
        print("  - data/test.csv")
        return
    
    # Step 3: Exploratory Data Analysis
    print("\n STEP 3: Exploratory Data Analysis...")
    explore_data(train_df, "Training Data")
    
    # Visualize survival distribution
    plot_survival_distribution(train_df)
    
    # Step 4: Initialize and Configure Preprocessor
    print("\n STEP 4: Preprocessing data...")
    preprocessor = DataPreprocessor(train_df)
    
    # Prepare features
    X = preprocessor.prepare_features(train_df, is_training=True)
    y = train_df['Survived']
    
    print(f"\n Features prepared:")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target shape: {y.shape}")
    
    # Step 5: Split Data
    print("\n  STEP 5: Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    
    # Step 6: Scale Features
    print("\n  STEP 6: Scaling features...")
    X_train_scaled = preprocessor.scale_features(X_train, fit=True)
    X_val_scaled = preprocessor.scale_features(X_val, fit=False)
    
    # Step 7: Train Model
    print("\n STEP 7: Training Random Forest model...")
    model = TitanicModel(n_estimators=100, random_state=42)
    model.train(X_train_scaled, y_train)
    
    # Step 8: Evaluate Model
    print("\n STEP 8: Evaluating model performance...")
    results = model.evaluate(X_val_scaled, y_val)
    
    # Display feature importance
    model.display_feature_importance(top_n=10)
    
    # Step 9: Prepare Test Data
    print("\n STEP 9: Making predictions on test data...")
    X_test = preprocessor.prepare_features(test_df, is_training=False)
    X_test_scaled = preprocessor.scale_features(X_test, fit=False)
    
    # Step 10: Make Predictions
    test_predictions = model.predict(X_test_scaled)
    
    # Step 11: Save Results
    print("\n STEP 10: Saving results...")
    submission = save_submission(test_df['PassengerId'], test_predictions, 
                                'submissions/submission.csv')
    
    # Step 12: Save Model and Scaler
    print("\n STEP 11: Saving model artifacts...")
    model.save_model('models/random_forest_model.pkl')
    preprocessor.save_scaler('models/scaler.pkl')
    
    # Final Summary
    print("\n" + "="*60)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n FINAL RESULTS:")
    print(f"   Validation Accuracy: {results['accuracy']:.4f}")
    print(f"   Test Predictions: {len(test_predictions)} passengers")
    print(f"   Predicted Survivors: {test_predictions.sum()} ({test_predictions.mean()*100:.1f}%)")
    print(f"\n OUTPUT FILES:")
    print(f"   - submissions/submission.csv (Kaggle submission)")
    print(f"   - models/random_forest_model.pkl (Trained model)")
    print(f"   - models/scaler.pkl (Feature scaler)")
    print("\n Ready for Kaggle submission!")

if __name__ == "__main__":
    run_pipeline()