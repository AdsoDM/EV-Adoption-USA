import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from scipy import stats
import os

# Import custom functions
# Ensure adhoc_functions is in path
try:
    from adhoc_functions import plot_confidence_intervals
except ImportError:
    # Fallback if running relative
    import sys
    sys.path.append(os.path.dirname(__file__))
    from adhoc_functions import plot_confidence_intervals

def code_party(df):
    """Encodes Party column: Republican=0, Democratic=1"""
    if 'Party' in df.columns:
         party_map = {'Republican': 0, 'Democratic': 1}
         # Handle if it's already numeric or different string format
         df['Party'] = df['Party'].map(party_map).fillna(df['Party'])
         df['Party'] = pd.to_numeric(df['Party'], errors='coerce')
    return df

def run_pca_and_plots(df_numeric, title_suffix=""):
    """
    Runs PCA on numeric dataframe and produces standard plots.
    Returns pca object, scaled data, and pca_df.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    # DataFrame for plotting
    pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    pca_df.index = df_numeric.index
    
    explained_variance = pca.explained_variance_ratio_
    
    # 1. Scree Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 11), explained_variance[:10] * 100, alpha=0.7, label='Individual')
    plt.ylabel('Explained variance ratio (%)')
    plt.xlabel('Principal component')
    plt.title(f'Scree Plot {title_suffix}')
    plt.tight_layout()
    plt.show() # In script execution this might just display
    
    # 2. Variable Loadings (PC1 vs PC2)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    plt.figure(figsize=(8, 8))
    for i, feature in enumerate(df_numeric.columns):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, color='g', ha='center', va='center', fontsize=9)
        
    circle = plt.Circle((0,0), 1, color='b', fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
    plt.title(f'PCA Variable Loadings {title_suffix}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 3. Individuals
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
    plt.title(f'PCA Individuals {title_suffix}')
    plt.grid(True)
    
    # Annotate a few points if not too many
    if len(pca_df) < 100:
        for idx in pca_df.index:
            plt.text(pca_df.loc[idx, 'PC1'], pca_df.loc[idx, 'PC2'], idx, fontsize=8, alpha=0.7)
            
    plt.tight_layout()
    plt.show()
    
    return pca, pca_df

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    ev_clean_01 = pd.read_csv(data_path)
    
    # Code party
    ev_clean_01 = code_party(ev_clean_01)
    
    # --- Analysis 1: Full Dataset (excl Index, year) ---
    print("\n--- Analysis 1: Full Dataset ---")
    ev_clean_02 = ev_clean_01.select_dtypes(include=[np.number]).copy()
    
    # Create row names: state_year
    row_names = ev_clean_01['state'] + "_" + ev_clean_01['year'].astype(str)
    ev_clean_02.index = row_names
    
    if 'Index' in ev_clean_02.columns: ev_clean_02.drop(columns=['Index'], inplace=True)
    if 'year' in ev_clean_02.columns: ev_clean_02.drop(columns=['year'], inplace=True)
    
    pca_1, pca_df_1 = run_pca_and_plots(ev_clean_02, "(All Years)")
    
    # Regression PC2 ~ PC1
    # Note: Generally PC1 and PC2 are orthogonal by definition in PCA, so slope should be 0.
    # The R script does lm(PC2 ~ PC1), likely to demonstrate this or check something specific.
    X = sm.add_constant(pca_df_1['PC1'])
    model_pc = sm.OLS(pca_df_1['PC2'], X).fit()
    print("\nRegression PC2 ~ PC1:")
    print(model_pc.summary())
    
    # Regression EV.Share ~ PC1 + PC2
    # Check if 'EV.Share....' exists. In clean csv it might be named differently?
    # Usually it's 'EV Share' or similar. Let's check columns if error.
    target_col = 'EV.Share....'
    if target_col not in ev_clean_01.columns:
        # Try to find similar
        candidates = [c for c in ev_clean_01.columns if "EV" in c and "Share" in c]
        if candidates:
            target_col = candidates[0]
            print(f"Using target column: {target_col}")
        else:
             print("Target EV Share column not found.")
             target_col = None
             
    if target_col:
        y = ev_clean_01[target_col].values
        # Match indices
        X_reg = sm.add_constant(pca_df_1[['PC1', 'PC2']])
        model_ev = sm.OLS(y, X_reg).fit()
        print(f"\nRegression {target_col} ~ PC1 + PC2:")
        print(model_ev.summary())
        
    # --- Analysis 2: Year 2023 ---
    print("\n--- Analysis 2: Year 2023 ---")
    ev_clean_03 = ev_clean_01[ev_clean_01['year'] == 2023].copy()
    ev_clean_03.set_index('state', inplace=True)
    
    # Select numeric
    ev_clean_03_num = ev_clean_03.select_dtypes(include=[np.number]).copy()
    
    # Drop columns as per R script: Index, year, fuel_economy
    drop_cols = ['Index', 'year', 'fuel_economy']
    ev_clean_03_num.drop(columns=[c for c in drop_cols if c in ev_clean_03_num.columns], inplace=True)
    
    pca_2, pca_df_2 = run_pca_and_plots(ev_clean_03_num, "(2023)")
    
    # Plot with Party coloring
    plt.figure(figsize=(10, 8))
    # Map Party back to colors
    # ev_clean_03 contains Party col (0 or 1)
    party_colors = {0: 'red', 1: 'blue'}
    colors = ev_clean_03['Party'].map(party_colors)
    
    plt.scatter(pca_df_2['PC1'], pca_df_2['PC2'], c=colors, alpha=0.6)
    
    # Add ellipses if possible (seaborn/matplotlib need manual work or specific lib). 
    # Skipping ellipses for simplicity, focus on points.
    
    # Label states
    for idx in pca_df_2.index:
        plt.text(pca_df_2.loc[idx, 'PC1'], pca_df_2.loc[idx, 'PC2'], idx, fontsize=8, alpha=0.7)
        
    plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA 2023 by Political Affiliation (Red=Rep, Blue=Dem)")
    plt.show()
    
    # --- Analysis 3: Remove EV Share from PCA input ---
    print("\n--- Analysis 3: PCA without EV Share (2023) ---")
    if target_col in ev_clean_03_num.columns:
        ev_clean_04 = ev_clean_03_num.drop(columns=[target_col])
    else:
        ev_clean_04 = ev_clean_03_num.copy()
        
    pca_3, pca_df_3 = run_pca_and_plots(ev_clean_04, "(2023, No EV Share)")
    
    # Regression EV Share ~ PC1
    if target_col:
        y_2023 = ev_clean_03[target_col]
        data_reg = pd.DataFrame({'Objective': y_2023, 'PC1': pca_df_3['PC1'], 'PC2': pca_df_3['PC2']}, index=ev_clean_03.index)
        
        X_reg3 = sm.add_constant(data_reg['PC1'])
        model_pca3 = sm.OLS(data_reg['Objective'], X_reg3).fit()
        print("\nRegression EV Share ~ PC1 (2023, PCA without EV Share):")
        print(model_pca3.summary())
        
        # Plot with confidence intervals
        try:
            plot_confidence_intervals(model_pca3, data_reg, x_col='PC1', y_col='Objective')
            plt.show()
        except Exception as e:
            print(f"Could not plot confidence intervals: {e}")
            
        # Diagnostics
        print("\n--- Model Diagnostics ---")
        # Normality
        shapiro_test = stats.shapiro(model_pca3.resid)
        print(f"Shapiro-Wilk: {shapiro_test}")
        
    # --- Analysis 4: Remove Outlier 'California' ---
    print("\n--- Analysis 4: Remove California (2023) ---")
    if "California" in ev_clean_04.index:
        ev_clean_05 = ev_clean_04.drop("California")
        y_no_cal = ev_clean_03.loc[ev_clean_03.index != "California", target_col]
        
        pca_4, pca_df_4 = run_pca_and_plots(ev_clean_05, "(2023, No California)")
        
        # Regression
        data_reg_no_cal = pd.DataFrame({'Objective': y_no_cal, 'PC1': pca_df_4['PC1']}, index=ev_clean_05.index)
        
        X_reg4 = sm.add_constant(data_reg_no_cal['PC1'])
        model_pca4 = sm.OLS(data_reg_no_cal['Objective'], X_reg4).fit()
        print("\nRegression EV Share ~ PC1 (2023, No California):")
        print(model_pca4.summary())
        
        try:
            plot_confidence_intervals(model_pca4, data_reg_no_cal, x_col='PC1', y_col='Objective')
            plt.title("Regression without California")
            plt.show()
        except Exception as e:
            print(f"Could not plot CI: {e}")

if __name__ == "__main__":
    main()
