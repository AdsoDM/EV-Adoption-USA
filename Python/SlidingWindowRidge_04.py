import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def main():
    # Load the dataset
    # Considering the script is in 'Python' folder and data might be in '..' (root of project) or just generated
    # The previous cleaning script saved it to os.path.join(os.path.dirname(__file__), "..", "ev_adoption_clean.csv")
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}. Checking current directory...")
        data_path = "ev_adoption_clean.csv"
        if not os.path.exists(data_path):
             print("Error: ev_adoption_clean.csv not found.")
             return

    df = pd.read_csv(data_path)

    # Feature selection based on theoretical relevance and previous discussion
    target_col = 'EV Share (%)'
    features = [
        'Stations', 'Incentives', 'Per_Cap_Income', 
        'gasoline_price_per_gallon', 'Unemployment_Rate',
        'Labour_Force_Participation_Rate', 'Bachelor_Attainment'
    ]
    
    # Check if columns exist
    missing_cols = [c for c in features + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return

    # --- 1. Visual Exploratory Data Analysis ---
    plt.figure(figsize=(14, 5))

    # Check distribution shift after log1p
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_col], kde=True, color='teal')
    plt.title(r'Original Distribution of $EV\ Share\ (\%)$')

    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(df[target_col]), kde=True, color='darkorange')
    plt.title(r'Log-transformed Distribution: $\ln(1 + x)$')

    plt.tight_layout()
    output_img_path = os.path.join(base_dir, 'distribution_analysis.png')
    plt.savefig(output_img_path)
    print(f"Distribution plot saved to {output_img_path}")

    # --- 2. Robust Modeling with Sliding Window ---
    window_size = 3
    years = sorted(df['year'].unique())
    backtesting_results = []
    
    print(f"Starting Sliding Window Backtesting over years: {years}")

    # Sliding window loop: train on 3 years, predict the next one
    for i in range(len(years) - window_size):
        train_years = years[i : i + window_size]
        test_year = years[i + window_size]
        
        # Filter data and handle potential missing values
        train_df = df[df['year'].isin(train_years)].dropna(subset=features + [target_col])
        test_df = df[df['year'] == test_year].dropna(subset=features + [target_col])
        
        if test_df.empty:
            print(f"Skipping test year {test_year} due to empty test set found (possibly missing data).")
            continue
            
        X_train = train_df[features]
        y_train = np.log1p(train_df[target_col]) # Log-transform target
        
        X_test = test_df[features]
        y_test = np.log1p(test_df[target_col])
        
        # Scale features: Crucial for Ridge Regularization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit Ridge Regression (L2 penalty)
        # alpha=1.0 is the regularization strength (lambda)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train_scaled, y_train)
        
        # Make predictions and reverse log transform for evaluation
        log_preds = ridge_model.predict(X_test_scaled)
        orig_preds = np.expm1(log_preds)
        orig_actuals = np.expm1(y_test)
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(orig_actuals, orig_preds))
        mae = mean_absolute_error(orig_actuals, orig_preds)
        
        # Get feature importance (coefficients)
        coeffs = dict(zip(features, ridge_model.coef_))
        
        print(f"trained on {train_years} -> testing on {test_year} | RMSE: {rmse:.4f}")

        result_entry = {
            'Test_Year': test_year,
            'RMSE': rmse,
            'MAE': mae,
            'Intercept': ridge_model.intercept_
        }
        # Add coefficients to result
        result_entry.update(coeffs)
        
        backtesting_results.append(result_entry)

    # Export results for further analysis
    if backtesting_results:
        results_df = pd.DataFrame(backtesting_results)
        output_csv_path = os.path.join(base_dir, 'sliding_window_ridge_results.csv')
        results_df.to_csv(output_csv_path, index=False)
        
        print("\nBacktesting Evaluation:")
        # Display main metrics
        print(results_df[['Test_Year', 'RMSE', 'MAE']])
        print(f"\nResults saved to {output_csv_path}")

        # --- 3. Error Analysis and Visualization (Requested Update) ---
        results = results_df.copy()
        # Adapt column name for the plotting snippet
        results.rename(columns={'Test_Year': 'Year'}, inplace=True)
        
        # Calculate the gap between RMSE and MAE
        results['Error_Gap'] = results['RMSE'] - results['MAE']

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Primary axis: Absolute Errors
        ax1.plot(results['Year'], results['RMSE'], marker='o', color='crimson', label='RMSE (Sensitivity to Outliers)')
        ax1.plot(results['Year'], results['MAE'], marker='s', color='navy', label='MAE (Average Error)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Absolute Error (EV Share %)')
        ax1.set_title('Backtesting Error Analysis')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Secondary axis: The Gap
        ax2 = ax1.twinx()
        ax2.bar(results['Year'], results['Error_Gap'], alpha=0.2, color='gray', label='RMSE-MAE Gap')
        ax2.set_ylabel('Sensitivity to Outliers (Gap)')
        ax2.legend(loc='upper right')

        output_plot_path = os.path.join(base_dir, 'error_interpretation.png')
        plt.savefig(output_plot_path)
        print(f"Error interpretation plot saved to {output_plot_path}")

        # Statistical Insight: Correlation between Error and Year
        if len(results) > 1:
            correlation = results['Year'].corr(results['RMSE'])
            print(f"Correlation between Year and RMSE: {correlation:.2f}")
    else:
        print("No results generated. Check data availability for sliding window years.")

if __name__ == "__main__":
    main()
