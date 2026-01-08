import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import os

def code_party(df):
    if 'Party' in df.columns:
         party_map = {'Republican': 0, 'Democratic': 1}
         df['Party'] = df['Party'].map(party_map).fillna(df['Party'])
         df['Party'] = pd.to_numeric(df['Party'], errors='coerce')
    return df

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "ev_adoption_clean.csv")
    
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    ev_clean = pd.read_csv(data_path)
    ev_clean = code_party(ev_clean)
    
    # --- Analysis 1: Global (All Years) ---
    print("\n--- Analysis 1: Global ---")
    ev_num = ev_clean.select_dtypes(include=[np.number]).copy()
    if 'Index' in ev_num.columns: ev_num.drop(columns=['Index'], inplace=True)
    if 'year' in ev_num.columns: ev_num.drop(columns=['year'], inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ev_num)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans k=2
    kmeans = KMeans(n_clusters=2, random_state=123, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    # Visualization simplified
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title("K-Means (k=2) on Global PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
    
    # --- Analysis 2: 2023 (with California) ---
    print("\n--- Analysis 2: 2023 ---")
    ev_2023 = ev_clean[ev_clean['year'] == 2023].copy()
    ev_2023.set_index('state', inplace=True)
    
    ev_2023_num = ev_2023.select_dtypes(include=[np.number]).copy()
    drop_cols = ['Index', 'year', 'fuel_economy']
    ev_2023_num.drop(columns=[c for c in drop_cols if c in ev_2023_num.columns], inplace=True)
    
    X_scaled_23 = scaler.fit_transform(ev_2023_num)
    X_pca_23 = pca.fit_transform(X_scaled_23)
    
    kmeans_23 = KMeans(n_clusters=2, random_state=123, n_init=10)
    clusters_23 = kmeans_23.fit_predict(X_pca_23)
    
    # --- Analysis 3: 2023 (No California) ---
    print("\n--- Analysis 3: 2023 (No California) ---")
    if "California" in ev_2023_num.index:
        ev_2023_no_cal = ev_2023.loc[ev_2023.index != "California"].copy()
        ev_2023_num_no_cal = ev_2023_num.drop("California")
        
        # PCA
        X_scaled_nc = scaler.fit_transform(ev_2023_num_no_cal)
        X_pca_nc = pca.fit_transform(X_scaled_nc)
        
        # KMeans
        kmeans_nc = KMeans(n_clusters=2, random_state=123, n_init=10)
        clusters_nc = kmeans_nc.fit_predict(X_pca_nc)
        
        # Prepare Plot Data
        plot_data = pd.DataFrame(X_pca_nc, columns=['PC1', 'PC2'], index=ev_2023_num_no_cal.index)
        plot_data['Cluster'] = clusters_nc.astype(str)
        plot_data['Party_Code'] = ev_2023_no_cal['Party']
        
        # Party Labels
        party_labels = {0: 'Republican', 1: 'Democrat'}
        plot_data['Party'] = plot_data['Party_Code'].map(party_labels)
        
        # Plot similar to R final plot
        plt.figure(figsize=(10, 8))
        
        # Colors: Party
        party_palette = {'Republican': 'red', 'Democrat': 'blue'}
        
        sns.scatterplot(
            data=plot_data, 
            x='PC1', y='PC2', 
            hue='Party', style='Cluster',
            palette=party_palette,
            s=100, alpha=0.8
        )
        
        # Add labels
        for idx in plot_data.index:
            plt.text(plot_data.loc[idx, 'PC1'], plot_data.loc[idx, 'PC2'], idx, fontsize=8, color='black', alpha=0.7)
            
        # Draw ellipses? Seaborn doesn't support ellipses natively in scatterplot easily.
        # Can use standard ellipse drawing if needed, but keeping it simple for now as requested by user constraints (minimal external libs).
        
        plt.title("Cluster Analysis vs Political Affiliation (No CA)")
        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(0, color='grey', linestyle='--')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
