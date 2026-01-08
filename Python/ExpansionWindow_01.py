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
    
    # Prepare data: numeric only
    ev_num = ev_clean.select_dtypes(include=[np.number]).copy()
    
    # Remove Index, fuel_economy (as per R script)
    # The R script selects numeric then removes Index, fuel_economy
    drop_cols = ['Index', 'fuel_economy']
    ev_num.drop(columns=[c for c in drop_cols if c in ev_num.columns], inplace=True)
    
    # Row names logic in R: paste state and year. In Python we can just filter by year column.
    # Note: R script keeps 'row.names' but doesn't seem to use them for logic other than identification?
    
    prediction_years = range(2017, 2023) # 2017 to 2022 inclusive
    
    rmse_results = []
    
    print("\n--- Expanding Window Analysis ---")
    
    for current_year in prediction_years:
        print(f"\nProcessing prediction for year: {current_year}")
        
        # Training years: 2016 to current_year - 1
        training_years = range(2016, current_year)
        
        training_data = ev_num[ev_num['year'].isin(training_years)].copy()
        test_data = ev_num[ev_num['year'] == current_year].copy()
        
        print(f"Training with {len(training_data)} observations from {min(training_years)} to {max(training_years)}")
        print(f"Testing on {len(test_data)} observations from year {current_year}")
        
        target_col = 'EV.Share....' 
        if target_col not in ev_num.columns:
             # Find it
             cands = [c for c in ev_num.columns if "EV" in c and "Share" in c]
             if cands: target_col = cands[0]
        
        # X and y
        y_train = training_data[target_col].values
        X_train = training_data.drop(columns=[target_col, 'year'])
        
        y_test = test_data[target_col].values
        X_test = test_data.drop(columns=[target_col, 'year'])
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PCA
        # R script uses 3 components
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Regression
        # R: lm(EV_Share ~ PC1 + PC2 + PC3)
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        
        # Predict
        predictions = model.predict(X_test_pca)
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE for {current_year}: {rmse:.5f}")
        rmse_results.append(rmse)
        
    # Final Results
    final_df = pd.DataFrame({
        'Predicted_Year': prediction_years,
        'RMSE': rmse_results
    })
    
    print("\n=========================================================")
    print("          Expanding Window Cross-Validation Results        ")
    print("=========================================================\n")
    print(final_df)
    print(f"\nAverage Model Performance (Mean RMSE): {np.mean(rmse_results):.5f}")
    print("=========================================================")

if __name__ == "__main__":
    main()
