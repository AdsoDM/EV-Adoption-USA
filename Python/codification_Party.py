import pandas as pd

def codify_party(df, col_name='Party', drop_first=True):
    """
    Converts a categorical column (default 'Party') into dummy/indicator variables.
    
    Equivalent to R's:
    dummy_cols(data, select_columns = "Party", remove_first_dummy = TRUE, remove_selected_columns = TRUE)
    """
    if col_name not in df.columns:
        print(f"Warning: Column '{col_name}' not found in DataFrame.")
        return df

    # pd.get_dummies with drop_first=True handles remove_first_dummy=TRUE
    # We concatenate with the original df (dropping the original column if desired)
    # The R script says remove_selected_columns = TRUE
    
    dummies = pd.get_dummies(df[col_name], prefix=col_name, drop_first=drop_first)
    
    # Convert bool to int (0/1) if needed, though pandas might might give bools
    dummies = dummies.astype(int)
    
    # Concatenate
    df_out = pd.concat([df.drop(columns=[col_name]), dummies], axis=1)
    
    return df_out

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({'Party': ['Democrat', 'Republican', 'Democrat', 'Independent', 'Republican']})
    print("Original:")
    print(data)
    print("\nProcessed:")
    print(codify_party(data))
