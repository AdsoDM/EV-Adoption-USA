import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    base_dir = os.path.dirname(__file__)
    
    # Paths to the result files
    ridge_path = os.path.join(base_dir, 'sliding_window_ridge_results.csv')
    rf_path = os.path.join(base_dir, 'sliding_window_rf_results.csv')
    
    # Check if files exist to use dynamic data
    if os.path.exists(ridge_path) and os.path.exists(rf_path):
        print(f"Loading results from files:\n - {ridge_path}\n - {rf_path}")
        
        # Load Ridge Results
        df_ridge = pd.read_csv(ridge_path)
        # Ridge CSV has 'Test_Year', 'RMSE', 'MAE', etc.
        # We need 'Test_Year' and 'RMSE'. Rename 'RMSE' to 'RMSE_Ridge'
        df_ridge = df_ridge[['Test_Year', 'RMSE']].rename(columns={'RMSE': 'RMSE_Ridge'})
        
        # Load RF Results
        df_rf = pd.read_csv(rf_path)
        # RF CSV has 'Test_Year', 'RMSE_RF', 'MAE_RF'
        df_rf = df_rf[['Test_Year', 'RMSE_RF']]
        
        # Merge
        results = pd.merge(df_ridge, df_rf, on='Test_Year', how='inner')
        results.rename(columns={'Test_Year': 'Year'}, inplace=True)
        
    else:
        print("Warning: Result files not found. Using hardcoded fallback data (from user request).")
        # Fallback values provided by user
        results = pd.DataFrame({
            'Year': [2019, 2020, 2021, 2022, 2023],
            'RMSE_Ridge': [0.1017, 0.1317, 0.1590, 0.3587, 0.4251],
            'RMSE_RF': [0.1101, 0.1394, 0.1930, 0.4096, 0.4708]
        })

    # Sort just in case
    results.sort_values('Year', inplace=True)

    # Calculate percentage difference
    # Positive means RF has higher error (worse)
    results['Diff_Pct'] = ((results['RMSE_RF'] - results['RMSE_Ridge']) / results['RMSE_Ridge']) * 100

    print("Comparison Data:")
    print(results)

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Ridge
    plt.plot(results['Year'], results['RMSE_Ridge'], 'o-', label='Ridge (Robust Linear)', color='#2c3e50', linewidth=2)
    
    # Random Forest
    plt.plot(results['Year'], results['RMSE_RF'], 's--', label='Random Forest (Non-linear)', color='#e67e22', linewidth=2)

    # Annotate differences
    for i in range(len(results)):
        year = results['Year'].iloc[i]
        rmse_rf = results['RMSE_RF'].iloc[i]
        diff = results['Diff_Pct'].iloc[i]
        
        # Determine sign
        sign_str = "+" if diff > 0 else ""
        
        plt.annotate(f"{sign_str}{diff:.1f}% error", 
                     (year, rmse_rf), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center', 
                     fontsize=9, 
                     color='#d35400')

    plt.title('Model Performance Comparison: Stability vs. Complexity', fontsize=14)
    plt.ylabel('RMSE (Lower is Better)')
    plt.xlabel('Test Year')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img_path = os.path.join(base_dir, 'performance_comparison.png')
    plt.savefig(output_img_path)
    print(f"Plot saved to {output_img_path}")

    print("\nRelative difference (Random Forest vs Ridge):")
    print(results[['Year', 'Diff_Pct']])

    # --- Feature Importance Analysis (Requested Update) ---
    print("\nStarting Feature Importance Analysis (Full Model)...")
    from sklearn.ensemble import RandomForestRegressor
    
    # Load full dataset for importance analysis
    clean_data_path = ridge_path.replace('sliding_window_ridge_results.csv', '..\\ev_adoption_clean.csv') # Fallback logic
    if not os.path.exists(clean_data_path):
        clean_data_path = 'ev_adoption_clean.csv'
        if not os.path.exists(clean_data_path):
             # Try relative path one level up if script is in specific folder
             clean_data_path = os.path.join(base_dir, '..', 'ev_adoption_clean.csv')

    if os.path.exists(clean_data_path):
        df = pd.read_csv(clean_data_path)
        target = 'EV Share (%)'
        features = ['Stations', 'Incentives', 'Per_Cap_Income', 'gasoline_price_per_gallon', 
                    'Unemployment_Rate', 'Labour_Force_Participation_Rate', 'Bachelor_Attainment']

        df_clean = df.dropna(subset=features + [target])
        X = df_clean[features]
        y = np.log1p(df_clean[target])

        # Train Model
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt')
        rf.fit(X, y)

        # Extract and Plot Importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 6))
        plt.title('Why is the model "Floating"? - Feature Importance')
        plt.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importance (Mean Decrease in Impurity)')
        plt.tight_layout()
        
        fi_output_path = os.path.join(base_dir, 'feature_importance_analysis.png')
        plt.savefig(fi_output_path)
        print(f"Feature Importance plot saved to {fi_output_path}")
        
    else:
        print(f"Could not find data at {clean_data_path} for Feature Importance analysis.")


if __name__ == "__main__":
    main()
