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
    
    # --- Filter 2022 (Training) ---
    print("\n--- Training on 2022 ---")
    df_train = ev_clean[ev_clean['year'] == 2022].copy()
    
    if df_train.empty:
        print("No data for 2022 found.")
        return

    # Prepare features
    drop_cols = ['Index', 'year', 'fuel_economy', target_col]
    features_train = df_train.select_dtypes(include=[np.number]).copy()
    features_train.drop(columns=[c for c in drop_cols if c in features_train.columns], inplace=True)
    
    # Std > 0
    std = features_train.std()
    cols_to_keep = std[std > 0].index
    features_train = features_train[cols_to_keep]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_train)
    
    # PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    print(f"Explained Variance: {pca.explained_variance_ratio_[:5]}")
    
    # Regression using PC1 + PC2
    pca_df = pd.DataFrame(X_train_pca[:, :2], columns=['PC1', 'PC2'], index=df_train.index)
    y_train = df_train[target_col].values
    
    X_reg = sm.add_constant(pca_df)
    model = sm.OLS(y_train, X_reg).fit()
    print(model.summary())
    
    # --- Predict 2023 (Test) ---
    print("\n--- Predicting 2023 ---")
    df_test = ev_clean[ev_clean['year'] == 2023].copy()
    
    if df_test.empty:
        print("No data for 2023 found.")
        return

    # Prepare features 2023
    # Must use same columns as training
    features_test = df_test[features_train.columns].copy()
    
    # Transform
    X_test_scaled = scaler.transform(features_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    pca_test_df = pd.DataFrame(X_test_pca[:, :2], columns=['PC1', 'PC2'], index=df_test.index)
    
    # Predict
    X_test_reg = sm.add_constant(pca_test_df)
    y_pred = model.predict(X_test_reg)
    y_actual = df_test[target_col].values
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    print(f"RMSE (2022 model on 2023 Data): {rmse}")
    
    sd_test = np.std(y_actual, ddof=1)
    print(f"Std Dev of EV Share 2023: {sd_test}")
    
    # Baseline Model (Mean of 2022)
    mean_train = np.mean(y_train)
    rmse_baseline = np.sqrt(np.mean((y_actual - mean_train)**2))
    print(f"RMSE Baseline (Mean 2022): {rmse_baseline}")

if __name__ == "__main__":
    main()
