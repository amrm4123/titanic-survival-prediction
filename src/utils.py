"""
Utility Functions for Titanic Project
Data loading, visualization, and helper functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def load_data(train_path: str = 'data/train.csv', 
              test_path: str = 'data/test.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Titanic datasets from CSV files.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV
    test_path : str
        Path to test data CSV
    
    Returns:
    --------
    tuple: (train_df, test_df) DataFrames
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f" Data loaded successfully:")
        print(f"   Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        print(f"   Test set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f" Error loading data: {e}")
        print(f"Please ensure files exist at:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        raise

def explore_data(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Perform basic exploratory data analysis and display results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to explore
    name : str
        Name of dataset for display
    """
    print(f"\n{'='*60}")
    print(f"EXPLORATORY DATA ANALYSIS: {name.upper()}")
    print(f"{'='*60}")
    
    # Basic information
    print(f"\n SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Data types
    print(f"\n DATA TYPES:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing values
    print(f"\n MISSING VALUES:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        print("  No missing values found!")
    else:
        for col, count in missing.items():
            percentage = (count / len(df)) * 100
            print(f"  {col:15}: {count:4} values ({percentage:.1f}%)")
    
    # Basic statistics for numerical columns
    print(f"\n BASIC STATISTICS (Numerical Columns):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats = df[numerical_cols].describe()
        print(stats.round(2))
    
    # Unique values for categorical columns
    print(f"\n CATEGORICAL VALUES:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        print(f"  {col:15}: {unique_vals:2} unique values")

def plot_survival_distribution(df: pd.DataFrame, figsize: tuple = (10, 6)) -> None:
    """
    Plot survival distribution with enhanced visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Survived' column
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Count survival
    survival_counts = df['Survived'].value_counts().sort_index()
    labels = ['Not Survived (0)', 'Survived (1)']
    colors = ['#ff6b6b', '#51cf66']
    
    # Create bar plot
    bars = plt.bar(labels, survival_counts.values, color=colors, edgecolor='black', linewidth=2)
    
    # Add count labels on bars
    for bar, count in zip(bars, survival_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Customize plot
    plt.title('Titanic Survival Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.xlabel('Survival Status', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n SURVIVAL SUMMARY:")
    print(f"   Total Passengers: {len(df)}")
    print(f"   Survived: {survival_counts.get(1, 0)} ({survival_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"   Not Survived: {survival_counts.get(0, 0)} ({survival_counts.get(0, 0)/len(df)*100:.1f}%)")

def save_submission(passenger_ids: pd.Series, 
                    predictions: np.ndarray, 
                    path: str = 'submissions/submission.csv') -> pd.DataFrame:
    """
    Save predictions in Kaggle submission format.
    
    Parameters:
    -----------
    passenger_ids : pd.Series
        Passenger IDs from test set
    predictions : np.ndarray
        Model predictions (0 or 1)
    path : str
        Path to save submission file
    
    Returns:
    --------
    pd.DataFrame: Submission DataFrame
    """
    # Ensure predictions are integers
    predictions = predictions.astype(int)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })
    
    # Save to CSV
    submission.to_csv(path, index=False)
    
    # Display summary
    print(f"\n SUBMISSION FILE CREATED:")
    print(f"   Saved to: {path}")
    print(f"   Total predictions: {len(submission)}")
    print(f"   Predicted survivors: {submission['Survived'].sum()} ({submission['Survived'].mean()*100:.1f}%)")
    print(f"   First 5 predictions:")
    print(submission.head().to_string(index=False))
    
    return submission

def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    """
    Plot correlation heatmap for numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing numerical features
    figsize : tuple
        Figure size (width, height)
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if len(numerical_df.columns) < 2:
        print("Not enough numerical columns for correlation heatmap.")
        return
    
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()