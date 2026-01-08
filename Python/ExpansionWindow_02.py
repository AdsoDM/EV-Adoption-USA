import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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
    
    ev_num = ev_clean.select_dtypes(include=[np.number]).copy()
    drop_cols = ['Index', 'fuel_economy']
    ev_num.drop(columns=[c for c in drop_cols if c in ev_num.columns], inplace=True)
    
    prediction_years = range(2017, 2023) 
    
    rmse_results = []
    
    print("\n--- Expanding Window Analysis (With Year Predictor) ---")
    
    target_col = 'EV.Share....' 
    if target_col not in ev_num.columns:
             cands = [c for c in ev_num.columns if "EV" in c and "Share" in c]
             if cands: target_col = cands[0]

    for current_year in prediction_years:
        print(f"\nProcessing prediction for year: {current_year}")
        
        training_years = range(2016, current_year)
        
        training_data = ev_num[ev_num['year'].isin(training_years)].copy()
        test_data = ev_num[ev_num['year'] == current_year].copy()
        
        print(f"Training with {len(training_data)} observations")
        
        y_train = training_data[target_col].values
        # Drop target and year from PCA input, but keep year for regression
        X_train_pca_input = training_data.drop(columns=[target_col, 'year'])
        years_train = training_data['year'].values.reshape(-1, 1)
        
        y_test = test_data[target_col].values
        X_test_pca_input = test_data.drop(columns=[target_col, 'year'])
        years_test = test_data['year'].values.reshape(-1, 1)
        
        # Scale PCA inputs
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca_input)
        X_test_scaled = scaler.transform(X_test_pca_input)
        
        # PCA
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Regression: Features = PC1, PC2, PC3, Year
        # Combine
        X_train_final = np.hstack((X_train_pca, years_train))
        X_test_final = np.hstack((X_test_pca, years_test))
        
        model = LinearRegression()
        model.fit(X_train_final, y_train)
        
        predictions = model.predict(X_test_final)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE for {current_year}: {rmse:.5f}")
        rmse_results.append(rmse)
        
    final_df = pd.DataFrame({
        'Predicted_Year': prediction_years,
        'RMSE': rmse_results
    })
    
    print("\n=========================================================")
    print("          Expanding Window Results (with Year)           ")
    print("=========================================================\n")
    print(final_df)
    print(f"\nAverage Model Performance (Mean RMSE): {np.mean(rmse_results):.5f}")
    print("=========================================================")

if __name__ == "__main__":
    main()
