import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from adhoc_functions import (
    imputar_regresion_filtrada, 
    imputar_con_tendencia, 
    imputar_regresion, 
    getmode,
    print_atributes # Optional
)

np.random.seed(123)

def clean_dataset(df, is_training=True):
    # This function encapsulates the cleaning logic to be applied 
    # consistently. Note: In a strict ML pipeline, imputation stats (mean/mode/reg models)
    # should be fitted on training and applied to test. 
    # However, to closely mirror the R script structure where 02 and 03 are separate, 
    # we replicate the logic here.
    
    # Copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # 1. Fuel Economy - Regression Imputation
    # Note: R script used 2018, 2019, 2020 as model years.
    prediction_years_model = [2018, 2019, 2020]
    df['fuel_economy'] = imputar_regresion_filtrada(
        df['fuel_economy'].values, 
        df['year'].values, 
        prediction_years_model
    )
    
    # 2. Incentives - Mean Imputation
    # If strictly replicating R which might use global mean of the current set
    mean_incentives = df['Incentives'].mean()
    df['Incentives'].fillna(mean_incentives, inplace=True)
    
    # 3. Metro Committees - Specific logic for 2019
    def impute_metro(group):
        mean_val = group.loc[group['year'].isin([2018, 2020]), 'Number.of.Metro.Organizing.Committees'].mean()
        mask = (group['year'] == 2019) & (group['Number.of.Metro.Organizing.Committees'].isna())
        if mask.any():
            group.loc[mask, 'Number.of.Metro.Organizing.Committees'] = round(mean_val) if not pd.isna(mean_val) else 0 # Fallback
        return group

    if 'Number.of.Metro.Organizing.Committees' in df.columns:
        df = df.groupby('state', group_keys=False).apply(impute_metro)

    # 4. Hot-deck / Trend Imputations (similar to Data_cleaning_01)
    
    # Helper for simple hot-deck
    def hotdeck_impute_simple(dframe, variable, domain_var):
        def impute_group(g):
            donors = g[variable].dropna()
            if len(donors) == 0: return g
            missing = g[variable].isna()
            if missing.any():
                g.loc[missing, variable] = np.random.choice(donors, size=missing.sum(), replace=True)
            return g
        return dframe.groupby(domain_var, group_keys=False).apply(impute_group)

    # Apply to various cols
    for col in ['affectweather', 'discuss', 'exp', 'localofficials', 'personal', 'reducetax', 'worried']:
        if col in df.columns:
            df = hotdeck_impute_simple(df, col, 'state')

    # Trend imputations
    if 'devharm' in df.columns:
        def impute_devharm(g):
            g['devharm'] = imputar_con_tendencia(g['devharm'].values, g['year'].values, [2018, 2019], [2016, 2017])
            return g
        df = df.groupby('state', group_keys=False).apply(impute_devharm)
        
    if 'regulate' in df.columns:
        def impute_regulate(g):
            g['regulate'] = imputar_con_tendencia(g['regulate'].values, g['year'].values, [2018, 2019, 2020], [2016, 2017])
            return g
        df = df.groupby('state', group_keys=False).apply(impute_regulate)

    # 5. Regression Imputations
    for col in ['gasoline_price_per_gallon', 'Trucks']:
        def impute_reg(g, c=col):
            g[c] = imputar_regresion(g[c].values, g['year'].values)
            return g
        df = df.groupby('state', group_keys=False).apply(impute_reg)

    # 6. DC Imputation Strategy (KNN)
    # Only applies if DC is in the set
    if "District Of Columbia" in df['state'].values:
        # Prepare data for 2023 (or latest available in this set)
        data_2023 = df[df['year'] == 2023].copy()
        
        # Check if we have enough data for KNN
        if not data_2023.empty:
            features = ['Per_Cap_Income', 'Population_20_64', 'gasoline_price_per_gallon', 'Bachelor_Attainment', 'EV.Share....']
            # Ensure features exist
            available_features = [f for f in features if f in df.columns]
            
            sim_data = data_2023.set_index('state')[available_features].dropna()
            
            if "District Of Columbia" in sim_data.index and len(sim_data) > 5:
                scaler = StandardScaler()
                sim_data_std = pd.DataFrame(scaler.fit_transform(sim_data), index=sim_data.index, columns=sim_data.columns)
                
                dc_data = sim_data_std.loc[["District Of Columbia"]]
                others = sim_data_std.drop("District Of Columbia")
                
                knn = NearestNeighbors(n_neighbors=min(5, len(others)))
                knn.fit(others)
                distances, indices = knn.kneighbors(dc_data)
                
                neighbor_names = others.iloc[indices[0]].index.tolist()
                
                # Impute Trucks
                trucks_means = df[df['state'].isin(neighbor_names)].groupby('year')['Trucks'].mean()
                
                # Update DC
                for y in trucks_means.index:
                    mask = (df['state'] == "District Of Columbia") & (df['year'] == y) & (df['Trucks'].isna())
                    if mask.any():
                        df.loc[mask, 'Trucks'] = trucks_means[y]
                
                # Impute Party
                party_modes = df[df['state'].isin(neighbor_names)].groupby('year')['Party'].apply(lambda x: getmode(x))
                 for y in party_modes.index:
                    mask = (df['state'] == "District Of Columbia") & (df['year'] == y) & (df['Party'].isna())
                    if mask.any():
                        df.loc[mask, 'Party'] = party_modes[y]

    # 7. Final Party Mode Imputation
    def impute_party(g):
        mode_v = getmode(g['Party'], na_rm=True)
        g['Party'].fillna(mode_v, inplace=True)
        return g
    df = df.groupby('state', group_keys=False).apply(impute_party)
    
    # 8. Drop unnecessary
    cols_drop = ["Total", "Trucks_Share"]
    df.drop(columns=[c for c in cols_drop if c in df.columns], inplace=True)
    
    return df

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
    
    # Clean Training
    print("Cleaning Training Set...")
    train_clean = clean_dataset(train_set, is_training=True)
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "training_clean.csv")
    train_clean.to_csv(output_path, index=False)
    print(f"Training clean data saved to {output_path}")

if __name__ == "__main__":
    main()
