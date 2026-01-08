import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Import custom functions
from adhoc_functions import print_boxplots

def plot_correlation_matrix(df):
    """
    Plots the correlation matrix for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show() # In script, maybe save or just show

def code_party(df):
    # Recode Party: Republican -> 0, Democratic -> 1
    # Check if Party column exists
    if 'Party' in df.columns:
         # Use map for explicit replacement. Ensure it works if already numeric or string.
         # R: factor(Party, levels=c('Republican','Democratic'), labels=c(0,1))
         # This implies Republican=0, Democratic=1
         
         party_map = {'Republican': 0, 'Democratic': 1}
         # If values are already 0/1 or mixed, this might need care. 
         # Assuming strings based on R script
         df['Party'] = df['Party'].map(party_map).fillna(df['Party']) # fillna to keep original if not mapped (e.g. if already 0/1)
         
         # Convert to numeric
         df['Party'] = pd.to_numeric(df['Party'], errors='coerce')
         
    return df

def pca_analysis(df):
    """
    Performs PCA analysis and plots results.
    """
    # Select numeric, drop Index and year if present
    df_pca = df.select_dtypes(include=[np.number]).copy()
    cols_to_drop = ['Index', 'year']
    df_pca.drop(columns=[c for c in cols_to_drop if c in df_pca.columns], inplace=True)
    
    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_pca)
    
    # PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    # Summary
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nPCA Summary:")
    print(f"Explained Variance Ratio: {explained_variance[:10]}") # Show first 10
    
    # 1. Scree Plot (fviz_eig)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7, align='center', label='Individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), cumulative_variance * 100, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio (%)')
    plt.xlabel('Principal component')
    plt.title('Scree Plot')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # 2. Variable Loadings (Circle of Correlation counterpart)
    # fviz_pca_var
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    plt.figure(figsize=(10, 10))
    for i, feature in enumerate(df_pca.columns):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, color='g', ha='center', va='center')
        
    circle = plt.Circle((0,0), 1, color='b', fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
    plt.title('PCA - Variable Loadings (Circle of Correlation)')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # 3. Individuals (fviz_pca_ind)
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
    plt.title('PCA - Individuals')
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, "..", "training_clean.csv")
    test_path = os.path.join(base_dir, "..", "test_clean.csv")
    
    if not os.path.exists(train_path):
        print(f"File not found: {train_path}. Please run data cleaning scripts first.")
        return

    training_01 = pd.read_csv(train_path)
    test_01 = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    
    # Code Party
    training_01 = code_party(training_01)
    if not test_01.empty:
        test_01 = code_party(test_01)
        
    # Dummy encoding for 'state'
    # R: training_coded <- dummy_cols(..., select_columns = "state", remove_first_dummy = TRUE)
    training_coded = pd.get_dummies(training_01, columns=['state'], drop_first=True, dtype=int)
    if not test_01.empty:
        test_coded = pd.get_dummies(test_01, columns=['state'], drop_first=True, dtype=int)
        
    # Correlation Matrix (on scaled data in R, but correlation is scale invariant if Pearson, 
    # though R scales first then correlates. We can just correlate numeric features.)
    # R scales then correlates. The correlation of scaled data is same as correlation of unscaled.
    plot_correlation_matrix(training_coded)
    
    # Boxplots
    # print_boxplots(training_01, "boxplots_py.pdf") # Uncomment to save
    # Note: Using path relative to script or parent?
    boxplot_path = os.path.join(base_dir, "..", "boxplots_py.pdf")
    try:
        print_boxplots(training_01, boxplot_path)
        print(f"Boxplots saved to {boxplot_path}")
    except Exception as e:
        print(f"Could not generate boxplots: {e}")

    # PCA
    pca_analysis(training_01)

if __name__ == "__main__":
    main()
