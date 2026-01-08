import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from Data_cleaning_02_training import clean_dataset 

# Re-use the clean_dataset function to ensure identical logic. 
# Ideally, "training" parameters (like means, regression models) should be saved from 02 and loaded here.
# But for now, we will perform the cleaning *procedure* on the test set. 
# Note carefully: Using 'imputar_regresion' on test set means we regress using test data points. 
# R script likely does this (independent cleaning) or 'Data_cleaning_03_test.R' is a copy-paste of 02.
# We will assume independent cleaning for now as per the file structure, 
# though correct ML practice would be transform(test) using fitted(train) params.

np.random.seed(123)

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "EV_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    ev_adoption = pd.read_csv(data_path)
    
    # Clean structure
    ev_adoption_01 = ev_adoption.iloc[:, 1:].copy()
    ev_adoption_01.rename(columns={ev_adoption_01.columns[0]: "Index"}, inplace=True)
    
    # Split
    train_set, test_set = train_test_split(ev_adoption_01, test_size=0.20, random_state=123)
    
    # Clean Test
    print("Cleaning Test Set...")
    test_clean = clean_dataset(test_set, is_training=False)
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "test_clean.csv")
    test_clean.to_csv(output_path, index=False)
    print(f"Test clean data saved to {output_path}")

if __name__ == "__main__":
    main()
