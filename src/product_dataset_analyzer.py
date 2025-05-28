import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config seaborn
plt.style.use('seaborn-v0_8')  
sns.set_theme() 

def load_product_data():
    """Load product data from csv file"""
    data_path = Path(__file__).parent.parent / 'data' / 'Product_Information_Dataset.csv'
    return pd.read_csv(data_path)

def print_column_info(df):
    """Print detailed information about the columns of the dataset."""
    print("\n=== Detailed Column Information ===")
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    print("\nData Types by Column:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    print("\nFirst rows of each column:")
    print(df.head())
    
    print("\nUnique values count by column:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")

def explore_dataset(df):
    """Basic exploratory analysis of the dataset."""
    print("\n=== General Dataset Information ===")
    print("\nDataset Dimensions:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Print detailed information about the columns
    print_column_info(df)
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nNull Values by Column:")
    print(df.isnull().sum())

def create_visualizations(df):
    """Create and save visualizations of the dataset."""
    # 1. Price Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='price', bins=50)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    # 2. Bar Chart of Count by Category (if exists)
    if 'main_category' in df.columns:
        plt.figure(figsize=(15, 6))
        category_counts = df['main_category'].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Number of Products by Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    
    # 3. Correlation Matrix (for numeric columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()

def main():
    # Load data
    print("Loading product dataset...")
    df = load_product_data()
    
    # Explore the dataset
    explore_dataset(df)
    

    create_visualizations(df)
    
    # Keep the visualizations open
    plt.show()

if __name__ == "__main__":
    main() 