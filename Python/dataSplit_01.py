import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, split_ratio=0.80, seed=123):
    """
    Splits the dataframe into training and test sets.
    
    Equivalent to R's:
    ksplit = sample.split(ev_adoption_01$Index, SplitRatio = 0.80)
    training_set = subset(..., ksplit==TRUE)
    test_set = subset(..., ksplit==FALSE)
    """
    # R's sample.split on a unique identifier (Index) effectively does a random split.
    # If the R code split on a target variable, it would be stratified.
    # Here we assume random split unless stratification is requested.
    
    test_size = 1.0 - split_ratio
    
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    
    return train_set, test_set

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({'Index': range(10), 'Value': range(10)})
    train, test = split_data(data)
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
