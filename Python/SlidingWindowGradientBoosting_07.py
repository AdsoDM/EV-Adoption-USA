import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Load data
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        data_path = 'ev_adoption_clean.csv'
        if not os.path.exists(data_path):
            print(f"Error: ev_adoption_clean.csv not found.")
            return

    df = pd.read_csv(data_path)

    target_col = 'EV Share (%)'
    features = [
        'Stations', 'Incentives', 'Per_Cap_Income', 
        'gasoline_price_per_gallon', 'Unemployment_Rate',
        'Labour_Force_Participation_Rate', 'Bachelor_Attainment'
    ]

    # Verify columns
    missing_cols = [c for c in features + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    # Sliding Window parameters
    window_size = 3
    years = sorted(df['year'].unique())
    gb_results = []

    print(f"Starting Gradient Boosting Sliding Window Backtesting over years: {years}")
    
    # We will store the last model to plot feature importance later
    last_gb_model = None

    for i in range(len(years) - window_size):
        train_years = years[i : i + window_size]
        test_year = years[i + window_size]
        
        train_df = df[df['year'].isin(train_years)].dropna(subset=features + [target_col])
        test_df = df[df['year'] == test_year].dropna(subset=features + [target_col])
        
        if test_df.empty: 
            print(f"Skipping test year {test_year} (no data)")
            continue
            
        X_train = train_df[features]
        y_train = np.log1p(train_df[target_col]) 
        
        X_test = test_df[features]
        y_test = np.log1p(test_df[target_col])
        
        # Initialize Gradient Boosting Regressor
        # Standard default parameters often work well, but we can tune them.
        # Starting with 100 estimators and 0.1 learning rate.
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_model.fit(X_train, y_train)
        last_gb_model = gb_model
        
        # Predict
        log_preds = gb_model.predict(X_test)
        orig_preds = np.expm1(log_preds)
        orig_actuals = np.expm1(y_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(orig_actuals, orig_preds))
        mae = mean_absolute_error(orig_actuals, orig_preds)
        
        print(f"Train {train_years} -> Test {test_year} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        gb_results.append({
            'Test_Year': test_year,
            'RMSE_GB': rmse,
            'MAE_GB': mae
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(gb_results)
    print("\nGradient Boosting Backtesting Evaluation:")
    print(results_df)

    # Save results
    output_path = os.path.join(base_dir, 'sliding_window_gb_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # --- Feature Importance Interpretation ---
    if last_gb_model is not None:
        importances = last_gb_model.feature_importances_
        indices = np.argsort(importances) # sort ascending

        plt.figure(figsize=(10, 6))
        plt.title('Gradient Boosting Feature Importance (Last Fold)')
        plt.barh(range(len(indices)), importances[indices], color='darkgreen', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        fi_output_path = os.path.join(base_dir, 'gb_feature_importance.png')
        plt.savefig(fi_output_path)
        print(f"Feature Importance plot saved to {fi_output_path}")

if __name__ == "__main__":
    main()
