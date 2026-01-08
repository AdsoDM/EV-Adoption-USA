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
    
    # Parameters
    window_size = 1
    start_year = 2016 + window_size
    prediction_years = range(start_year, 2023) 
    
    sliding_rmse_results = []
    baseline_rmse_results = []
    
    target_col = 'EV.Share....' 
    if target_col not in ev_num.columns:
             cands = [c for c in ev_num.columns if "EV" in c and "Share" in c]
             if cands: target_col = cands[0]

    print("\n--- Sliding Window Analysis ---")
    
    for current_year in prediction_years:
        print(f"\nProcessing prediction for year: {current_year}")
        
        # Training years: window_size years before current_year
        training_years = range(current_year - window_size, current_year)
        
        training_data = ev_num[ev_num['year'].isin(training_years)].copy()
        test_data = ev_num[ev_num['year'] == current_year].copy()
        
        print(f"Training with {len(training_data)} observations from {min(training_years)} to {max(training_years)}")
        print(f"Testing on {len(test_data)} observations from year {current_year}")
        
        y_train = training_data[target_col].values
        X_train = training_data.drop(columns=[target_col, 'year'])
        
        y_test = test_data[target_col].values
        X_test = test_data.drop(columns=[target_col, 'year'])
        
        # --- Baseline RMSE ---
        # Mean of training data
        mean_train = np.mean(y_train)
        baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, mean_train)))
        baseline_rmse_results.append(baseline_rmse)
        
        # --- PCA + Regression ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        predictions = model.predict(X_test_pca)
        
        model_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        sliding_rmse_results.append(model_rmse)
        
        print(f"  -> Baseline RMSE for {current_year}: {baseline_rmse:.5f}")
        print(f"  -> Model RMSE for    {current_year}: {model_rmse:.5f}")
        
    final_df = pd.DataFrame({
        'Predicted_Year': prediction_years,
        'Model_RMSE': sliding_rmse_results,
        'Baseline_RMSE': baseline_rmse_results
    })
    
    print("\n=========================================================")
    print("            Sliding Window - Final Results             ")
    print("=========================================================\n")
    print(final_df)
    print(f"\nAverage Model Performance (Mean RMSE):   {np.mean(sliding_rmse_results):.5f}")
    print(f"Average Baseline Performance (Mean RMSE): {np.mean(baseline_rmse_results):.5f}")
    print("=========================================================")

if __name__ == "__main__":
    main()
