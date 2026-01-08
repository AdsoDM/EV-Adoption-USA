import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import os

def code_party(df):
    if 'Party' in df.columns:
         party_map = {'Republican': 0, 'Democratic': 1}
         df['Party'] = df['Party'].map(party_map).fillna(df['Party'])
         df['Party'] = pd.to_numeric(df['Party'], errors='coerce')
    return df

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    ev_clean = pd.read_csv(data_path)
    ev_clean = code_party(ev_clean)
    
    target_col = 'EV.Share....'
    if target_col not in ev_clean.columns:
        cols = [c for c in ev_clean.columns if "EV" in c and "Share" in c]
        if cols: target_col = cols[0]
    
    # --- Filter 2016 (Training) ---
    print("\n--- Training on 2016 ---")
    df_2016 = ev_clean[ev_clean['year'] == 2016].copy()
    
    # Prepare features
    # Drop non-numeric + specific cols
    drop_cols = ['Index', 'year', 'fuel_economy', target_col]
    features_2016 = df_2016.select_dtypes(include=[np.number]).copy()
    features_2016.drop(columns=[c for c in drop_cols if c in features_2016.columns], inplace=True)
    
    # Handle std=0? 
    # R: "delete columns with standar deviation = 0" 
    # Logic: scaling requires std != 0
    std = features_2016.std()
    cols_to_keep = std[std > 0].index
    features_2016 = features_2016[cols_to_keep]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_2016)
    
    # PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Summary
    print(f"Explained Variance: {pca.explained_variance_ratio_[:5]}")
    
    # Regression using PC1 + PC2
    pca_df = pd.DataFrame(X_train_pca[:, :2], columns=['PC1', 'PC2'], index=df_2016.index)
    y_train = df_2016[target_col].values
    
    X_reg = sm.add_constant(pca_df)
    model = sm.OLS(y_train, X_reg).fit()
    print(model.summary())
    
    # --- Predict 2017 (Test) ---
    print("\n--- Predicting 2017 ---")
    df_2017 = ev_clean[ev_clean['year'] == 2017].copy()
    
    # Prepare features 2017
    # Must use same columns as 2016
    features_2017 = df_2017[features_2016.columns].copy()
    
    # Transform using 2016 scaler and PCA
    X_test_scaled = scaler.transform(features_2017)
    X_test_pca = pca.transform(X_test_scaled)
    
    pca_test_df = pd.DataFrame(X_test_pca[:, :2], columns=['PC1', 'PC2'], index=df_2017.index)
    
    # Predict
    X_test_reg = sm.add_constant(pca_test_df)
    # Ensure const is there
    y_pred = model.predict(X_test_reg)
    y_actual = df_2017[target_col].values
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"RMSE (2016 model on 2017 Data): {rmse}")
    
    sd_2017 = np.std(y_actual, ddof=1) # R uses sample sd (ddof=1)
    print(f"Std Dev of EV Share 2017: {sd_2017}")
    
    # Baseline Model (Mean of 2016)
    mean_2016 = np.mean(y_train)
    # RMSE baseline on 2017
    rmse_baseline = np.sqrt(np.mean((y_actual - mean_2016)**2))
    print(f"RMSE Baseline (Mean 2016): {rmse_baseline}")

if __name__ == "__main__":
    main()
