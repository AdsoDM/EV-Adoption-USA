import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
import os

# Import custom functions
# Assumes adhoc_functions.py is in the same directory or path is added
from adhoc_functions import (
    print_atributes, 
    imputar_regresion_filtrada, 
    imputar_con_tendencia, 
    imputar_regresion, 
    getmode
)

#Manage code paths
base_path_data = Path(r"E:\Oscar\adoptionUSA\EV-Adoption-USA\data")
data_file = base_path_data / "EV_Data.csv"

# Set seed for reproducibility
np.random.seed(123)

def main():
    # Load data
    # Assuming CSV is in the parent directory as per original code implication ".../EV_data.csv"
    # Adjust path as necessary relative to where this script runs
    data_path = data_file
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    ev_adoption = pd.read_csv(data_path)

    # Data Wrangling
    # Remove the first column as it is duplicated and rename the second to Index
    # Original R: ev_adoption_01 <- ev_adoption[,-1]; colnames(ev_adoption_01)[1]<- "Index"
    ev_adoption_01 = ev_adoption.iloc[:, 1:].copy()
    ev_adoption_01.rename(columns={ev_adoption_01.columns[0]: "Index"}, inplace=True)
    
    # Backup
    ev_adoption_01_BU = ev_adoption_01.copy()
    
    # Analyze missing values
    print("Total missing values:", ev_adoption_01.isna().sum().sum())
    print("Complete cases:", ev_adoption_01.dropna().shape[0])
    
    # Split data (Logic only - actual split for saving done later or in another script in R but here we follow the flow)
    # R code splits 80/20. 
    # ksplit = sample.split(ev_adoption_01$Index, SplitRatio = 0.80)
    train_set, test_set = train_test_split(ev_adoption_01, test_size=0.20, random_state=123, stratify=None)
    
    # Handling missing data
    # print_atributes(ev_adoption_01, output_path="Variable_Analysis_Py.pdf")
    
    # Imputation analysis - Count NAs per column
    conteo_nas_por_columna = ev_adoption_01.isna().sum()
    # print(conteo_nas_por_columna)
    
    # --- Variable Fuel economy ---
    # Impute using linear regression
    # R: imputar_regresion_filtrada(valores = fuel_economy, anios = year, anios_modelo = c(2018,2019,2020))
    
    # Group by state to apply imputation per state? R code does:
    # ev_adoption_01 <- ev_adoption_01 %>% mutate(fuel_economy = imputar_regresion_filtrada(...))
    # It does NOT seem to group by state in the R code for THIS variable on lines 107-115.
    # Checks line 102 says "Apply the regression ... to each state", but line 107 does NOT have a group_by(state).
    # Wait, looking at R code trace:
    # 107: ev_adoption_01 <- ev_adoption_01 %>% mutate(...) 
    # This implies it applies it to the WHOLE dataset, using 'year' as predictor. 
    # BUT line 117 says "Verify the result for a state".
    # And line 103 says "Apply ... to each state". 
    # If the R code missed the `group_by(state)`, then it did it globally.
    # However, for `devharm` (line 249) it explicitly does `group_by(state)`.
    # For `fuel_economy`, it seems to claim to do it per state but the code provided lacks `group_by`. 
    # Note: `fuel_economy` is described as "average consumption for the entire country...". 
    # So it makes sense it is NOT grouped by state! It's a national metric.
    
    prediction_years_model = [2018, 2019, 2020]
    
    # It seems `fuel_economy` is indeed global.
    ev_adoption_01['fuel_economy'] = imputar_regresion_filtrada(
        ev_adoption_01['fuel_economy'].values, 
        ev_adoption_01['year'].values, 
        prediction_years_model
    )
    
    # --- Incentives ---
    # Impute by mean
    # R: ifelse(is.na(Incentives), ave(Incentives, FUN = mean), Incentives)
    # ave() without grouping calculates global mean.
    mean_incentives = ev_adoption_01['Incentives'].mean()
    ev_adoption_01['Incentives'] = ev_adoption_01['Incentives'].fillna(mean_incentives)
    
    # --- Number.of.Metro.Organizing.Committees ---
    # Impute using mean of 2018, 2020 for year 2019, grouped by state
    
    def impute_metro(group):
        # Calculate mean for 2018, 2020
        mean_val = group.loc[group['year'].isin([2018, 2020]), 'Number of Metro Organizing Committees'].mean()
        
        # Apply logic: if year 2019 and is NA, replace
        mask = (group['year'] == 2019) & (group['Number of Metro Organizing Committees'].isna())
        if mask.any():
            group.loc[mask, 'Number of Metro Organizing Committees'] = round(mean_val)
        return group

    ev_adoption_01 = ev_adoption_01.groupby('state', group_keys=False).apply(impute_metro)
    
    # --- affectweather ---
    # Hot-deck imputation stratified by state
    # R uses VIM::hotdeck. Python doesn't have a direct VIM equivalent.
    # We will use our `imputar_con_tendencia` or a simple random sampling from donors in the same group (hot deck).
    # Since specific function `imputar_con_tendencia` was defined in adhoc, let's see if we use that.
    # R code at line 206 calls `hotdeck(...)`. This is from VIM package.
    # We need a custom hot-deck implementation.
    
    def hotdeck_impute_simple(df, variable, domain_var):
        # Simple random hot-deck within domain
        def impute_group(group):
            donors = group[variable].dropna()
            if len(donors) == 0:
                return group
            
            missing_mask = group[variable].isna()
            if missing_mask.any():
                imputed_values = np.random.choice(donors, size=missing_mask.sum(), replace=True)
                group.loc[missing_mask, variable] = imputed_values
            return group

        return df.groupby(domain_var, group_keys=False).apply(impute_group)

    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'affectweather', 'state')

    # --- devharm ---
    # Trend imputation (Hot-deck with years)
    # R code line 249: group_by(state) %>% mutate(devharm = imputar_con_tendencia(devharm, year, c(2018,2019), c(2016,2017)))
    
    def impute_devharm(group):
        group['devharm'] = imputar_con_tendencia(
            group['devharm'].values, 
            group['year'].values, 
            anios_donante=[2018, 2019], 
            anios_receptor=[2016, 2017]
        )
        return group

    ev_adoption_01 = ev_adoption_01.groupby('state', group_keys=False).apply(impute_devharm)
    
    # --- discuss ---
    # Hot-deck by state
    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'discuss', 'state')
    
    # --- exp ---
    # Hot-deck by state
    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'exp', 'state')
    
    # --- localofficials ---
    # Hot-deck by state
    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'localofficials', 'state')
    
    # --- personal ---
    # Hot-deck by state
    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'personal', 'state')
    
    # --- reducetax ---
    # Hot-deck by state
    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'reducetax', 'state')
    
    # --- regulate ---
    # Trend imputation 2018-2020 donors, 2016-2017 receptors
    def impute_regulate(group):
        group['regulate'] = imputar_con_tendencia(
            group['regulate'].values, 
            group['year'].values, 
            anios_donante=[2018, 2019, 2020], 
            anios_receptor=[2016, 2017]
        )
        return group
    
    ev_adoption_01 = ev_adoption_01.groupby('state', group_keys=False).apply(impute_regulate)
    
    # --- worried ---
    # Hot-deck by state
    ev_adoption_01 = hotdeck_impute_simple(ev_adoption_01, 'worried', 'state')
    
    # --- gasoline_price_per_gallon ---
    # Linear regression imputation per state
    def impute_gas(group):
        group['gasoline_price_per_gallon'] = imputar_regresion(
            group['gasoline_price_per_gallon'].values, 
            group['year'].values
        )
        return group
        
    ev_adoption_01 = ev_adoption_01.groupby('state', group_keys=False).apply(impute_gas)
    
    # --- Trucks ---
    # Linear regression imputation per state
    def impute_trucks(group):
        group['Trucks'] = imputar_regresion(
            group['Trucks'].values, 
            group['year'].values
        )
        return group
        
    ev_adoption_01 = ev_adoption_01.groupby('state', group_keys=False).apply(impute_trucks)
    
    # --- District of Columbia KNN Strategy ---
    # Steps 600+ in R file
    
    # 1. Prepare data for similarity (Year 2023)
    data_para_similitud = ev_adoption_01[ev_adoption_01['year'] == 2023].copy()
    features = ['Per_Cap_Income', 'Population_20_64', 'gasoline_price_per_gallon', 'Bachelor_Attainment', 'EV Share (%)']
    
    sim_data = data_para_similitud.set_index('state')[features]
    
    # 2. Standardize
    scaler = StandardScaler()
    sim_data_std = pd.DataFrame(scaler.fit_transform(sim_data), index=sim_data.index, columns=sim_data.columns)
    
    # 3. Separate DC
    if "District Of Columbia" in sim_data_std.index:
        dc_data_std = sim_data_std.loc[["District Of Columbia"]]
        otros_estados_std = sim_data_std.drop("District Of Columbia")
        
        # 4. Find K nearest neighbors (k=5)
        # sklearn NearestNeighbors
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(otros_estados_std)
        distances, indices = knn.kneighbors(dc_data_std)
        
        nombres_vecinos = otros_estados_std.iloc[indices[0]].index.tolist()
        print(f"Neighbors for DC: {nombres_vecinos}")
        
        # 5. Impute 'Trucks' for DC using mean of neighbors
        imputacion_por_anio = ev_adoption_01[ev_adoption_01['state'].isin(nombres_vecinos)].groupby('year')['Trucks'].mean().reset_index()
        imputacion_por_anio.rename(columns={'Trucks': 'trucks_imputado'}, inplace=True)
        
        # Merge and fill
        ev_adoption_01 = pd.merge(ev_adoption_01, imputacion_por_anio, on='year', how='left')
        
        mask_dc_trucks = (ev_adoption_01['state'] == "District Of Columbia") & (ev_adoption_01['Trucks'].isna())
        ev_adoption_01.loc[mask_dc_trucks, 'Trucks'] = ev_adoption_01.loc[mask_dc_trucks, 'trucks_imputado']
        ev_adoption_01.drop(columns=['trucks_imputado'], inplace=True)
        
        # 6. Impute 'Party' using mode of neighbors
        # R uses mode.
        imputacion_por_moda = ev_adoption_01[ev_adoption_01['state'].isin(nombres_vecinos)].groupby('year')['Party'].apply(lambda x: getmode(x)).reset_index()
        imputacion_por_moda.rename(columns={'Party': 'party_imputado'}, inplace=True)
        
        ev_adoption_01 = pd.merge(ev_adoption_01, imputacion_por_moda, on='year', how='left')
        mask_dc_party = (ev_adoption_01['state'] == "District Of Columbia") & (ev_adoption_01['Party'].isna())
        ev_adoption_01.loc[mask_dc_party, 'Party'] = ev_adoption_01.loc[mask_dc_party, 'party_imputado']
        ev_adoption_01.drop(columns=['party_imputado'], inplace=True)

    # --- Impute Party for all states (years 2016-2017) by Mode ---
    def impute_party_mode(group):
        mode_val = getmode(group['Party'], na_rm=True)
        group['Party'] = group['Party'].fillna(mode_val)
        return group

    ev_adoption_01 = ev_adoption_01.groupby('state', group_keys=False).apply(impute_party_mode)
    
    # --- Final Cleaning ---
    # Remove 'Total', 'Trucks_Share'
    cols_to_drop = ["Total", "Trucks_Share"]
    ev_adoption_01.drop(columns=[c for c in cols_to_drop if c in ev_adoption_01.columns], inplace=True)
    
    # Select specific range if needed, R selects Index:Party. 
    # Just ensuring we keep everything relevant.
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "..", "ev_adoption_clean.csv")
    ev_adoption_01.to_csv(output_path, index=False)
    print(f"Clean data saved to {output_path}")

if __name__ == "__main__":
    main()
