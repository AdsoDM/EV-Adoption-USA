import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
import os

def code_party(df):
    if 'Party' in df.columns:
         party_map = {'Republican': 0, 'Democratic': 1}
         df['Party'] = df['Party'].map(party_map).fillna(df['Party'])
         df['Party'] = pd.to_numeric(df['Party'], errors='coerce')
    return df

def calculate_yoy(group):
    # Sort by year
    group = group.sort_values('year')
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    # Exclude year from pct change
    numeric_cols = [c for c in numeric_cols if c != 'year']
    
    # Calculate YoY change manually to match R: (x - lag(x)) / (lag(x) + 1e-9)
    # This prevents division by zero
    lagged = group[numeric_cols].shift(1)
    
    # We can't use simple division for the whole dataframe at once cleanly with the scalar add
    # because of column alignment.
    
    diff = group[numeric_cols] - lagged
    denom = lagged + 1e-9
    
    yoy_change = diff / denom
    
    # Rename columns
    yoy_change.columns = [f"{c}_YoY_Pct_Change" for c in numeric_cols]
    
    # Combine with original identifiers (State, Year)
    # We keep 'year' and 'state' from original group
    result = pd.concat([group[['year', 'state']], yoy_change], axis=1)
    
    return result

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    ev_clean = pd.read_csv(data_path)
    ev_clean = code_party(ev_clean)
    
    # Prepare base data
    drop_cols = ['Index', 'fuel_economy']
    ev_prep = ev_clean.drop(columns=[c for c in drop_cols if c in ev_clean.columns])
    
    # Calculate YoY
    yoy_data = ev_prep.groupby('state', group_keys=False).apply(calculate_yoy)
    yoy_data.dropna(inplace=True) # First year will be NaN
    
    # Target
    # Find column ending with EV.Share...._YoY_Pct_Change
    target_cols = [c for c in yoy_data.columns if "EV" in c and "Share" in c and c.endswith("_YoY_Pct_Change")]
    if not target_cols:
         print("Target not found after YoY transformation.")
         # Fallback search
         print(f"Available cols: {yoy_data.columns}")
         return
    target_variable = target_cols[0]
    print(f"Target Variable: {target_variable}")
    
    # Sliding Window
    window_size = 3
    years = sorted(yoy_data['year'].unique())
    start_predict = min(years) + window_size
    prediction_years = [y for y in years if y >= start_predict]
    
    sliding_rmse = []
    baseline_rmse = []
    
    print("\n--- Sliding Window YoY Analysis ---")
    
    for current_year in prediction_years:
        print(f"\nProcessing prediction for year: {current_year}")
        
        train_years = range(current_year - window_size, current_year)
        
        train_data = yoy_data[yoy_data['year'].isin(train_years)].copy()
        test_data = yoy_data[yoy_data['year'] == current_year].copy()
        
        print(f"Training with {len(train_data)} observations")
        
        # Prepare X and y
        y_train = train_data[target_variable].values
        X_train = train_data.drop(columns=[target_variable, 'year', 'state'])
        
        y_test = test_data[target_variable].values
        X_test = test_data.drop(columns=[target_variable, 'year', 'state'])
        
        # Remove zero variance cols (R: nearZeroVar)
        # sklearn VarianceThreshold(threshold=0) removes zero var
        selector = VarianceThreshold(threshold=0)
        X_train_clean = selector.fit_transform(X_train)
        X_test_clean = selector.transform(X_test)
        
        # Baseline
        mean_train = np.mean(y_train)
        rmse_base = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, mean_train)))
        baseline_rmse.append(rmse_base)
        
        # PCA + Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        
        # PCA
        # Handle case where components > samples or features
        n_comp = min(3, X_train_scaled.shape[1], X_train_scaled.shape[0])
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        preds = model.predict(X_test_pca)
        
        rmse_model = np.sqrt(mean_squared_error(y_test, preds))
        sliding_rmse.append(rmse_model)
        
        print(f"  -> Baseline RMSE: {rmse_base:.5f}")
        print(f"  -> Model RMSE:    {rmse_model:.5f}")
        
    final_df = pd.DataFrame({
        'Predicted_Year': prediction_years,
        'Model_RMSE': sliding_rmse,
        'Baseline_RMSE': baseline_rmse
    })
    
    print("\n=========================================================")
    print("        Sliding Window YoY - Final Results")
    print("=========================================================\n")
    print(final_df)
    print(f"\nAverage Model RMSE:    {np.mean(sliding_rmse):.5f}")
    print(f"Average Baseline RMSE: {np.mean(baseline_rmse):.5f}")
    print("=========================================================")

if __name__ == "__main__":
    main()
