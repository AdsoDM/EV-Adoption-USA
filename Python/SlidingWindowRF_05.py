import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def main():
    # Load data
    # Adjusting path to look in the parent directory where ev_adoption_clean.csv is located
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        # Fallback to local execution if script is run from root
        data_path = 'ev_adoption_clean.csv'
        
    if not os.path.exists(data_path):
        print(f"Error: ev_adoption_clean.csv not found at {data_path}")
        return

    df = pd.read_csv(data_path)

    # Defining features (same as before for a fair comparison)
    target_col = 'EV Share (%)'
    features = [
        'Stations', 'Incentives', 'Per_Cap_Income', 
        'gasoline_price_per_gallon', 'Unemployment_Rate',
        'Labour_Force_Participation_Rate', 'Bachelor_Attainment'
    ]

    # Verify columns exist
    missing_cols = [c for c in features + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"Error: The following columns are missing in the dataset: {missing_cols}")
        return

    # Sliding Window parameters
    window_size = 3
    years = sorted(df['year'].unique())
    rf_results = []

    # Sliding window loop
    print(f"Starting Random Forest Sliding Window Backtesting over years: {years}")
    
    for i in range(len(years) - window_size):
        train_years = years[i : i + window_size]
        test_year = years[i + window_size]
        
        train_df = df[df['year'].isin(train_years)].dropna(subset=features + [target_col])
        test_df = df[df['year'] == test_year].dropna(subset=features + [target_col])
        
        if test_df.empty: 
            print(f"Skipping test year {test_year} (no data)")
            continue
            
        X_train = train_df[features]
        # In Random Forest, log transformation is often less critical than in Linear Regression,
        # but we'll keep it for consistency in comparison.
        y_train = np.log1p(train_df[target_col]) 
        
        X_test = test_df[features]
        y_test = np.log1p(test_df[target_col])
        
        # Initialize Random Forest Regressor
        # n_estimators: Number of trees in the forest
        # max_features: Number of features to consider when looking for the best split
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt')
        rf_model.fit(X_train, y_train)
        
        # Predict and reverse log transform
        log_preds = rf_model.predict(X_test)
        orig_preds = np.expm1(log_preds)
        orig_actuals = np.expm1(y_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(orig_actuals, orig_preds))
        mae = mean_absolute_error(orig_actuals, orig_preds)
        
        print(f"Train {train_years} -> Test {test_year} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        rf_results.append({
            'Test_Year': test_year,
            'RMSE_RF': rmse,
            'MAE_RF': mae
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(rf_results)
    print("\nRandom Forest Backtesting Evaluation:")
    print(results_df)

    # Feature Importance: One of the best tools in Random Forest
    # Note: This takes the feature importance from the LAST model trained in the loop.
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop Predictors (Feature Importance from last fold):")
    print(importances)

    # Allow saving results to CSV for comparison with Ridge
    output_path = os.path.join(base_dir, 'sliding_window_rf_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
