import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import random

def getmode(v, na_rm=False):
    """
    Calculate the mode of a dataset.
    """
    if na_rm:
        v = v[~pd.isna(v)]
    
    # Remove empty strings
    v = v[v != ""]
    
    if len(v) == 0:
        return np.nan
        
    return v.mode()[0]

def print_atributes(input_data, output_path="Variable_Analysis.pdf"):
    """
    Create a PDF with the graphical representation of the different variables of a dataset.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    umbral_valores_unicos = 5
    
    with PdfPages(output_path) as pdf:
        for nombre_columna in input_data.columns:
            datos_columna = input_data[nombre_columna]
            
            plt.figure(figsize=(10, 7))
            
            # Logic to decide the type of plot
            if pd.api.types.is_string_dtype(datos_columna) or pd.api.types.is_categorical_dtype(datos_columna) or pd.api.types.is_object_dtype(datos_columna):
                 # Bar chart for categorical/character
                sns.countplot(x=datos_columna, color="steelblue", alpha=0.8)
                plt.title(f"Distribution of {nombre_columna}")
                plt.xlabel(nombre_columna)
                plt.ylabel("Frequency")
                plt.xticks(rotation=45)
                
            elif pd.api.types.is_numeric_dtype(datos_columna):
                n_unicos = len(datos_columna.dropna().unique())
                
                if n_unicos < umbral_valores_unicos:
                     # Treat as categorical
                    sns.countplot(x=datos_columna, color="skyblue", alpha=0.8)
                    plt.title(f"Distribution of {nombre_columna}")
                    plt.xlabel(nombre_columna)
                    plt.ylabel("Frequency")
                else:
                    # Histogram for continuous
                    sns.histplot(datos_columna, bins=30, color="salmon", alpha=0.8, edgecolor="white")
                    plt.title(f"Distribution of {nombre_columna}")
                    plt.xlabel(nombre_columna)
                    plt.ylabel("Frequency")
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def imputar_con_tendencia(valores, year, anios_donante, anios_receptor):
    """
    Function to perform time-driven hot-deck imputation.
    """
    valores = np.array(valores, dtype=object) # Use object to allow mixed types if needed, or stick to numeric
    year = np.array(year)
    
    # Convert inputs to list/array if they aren't
    if not isinstance(anios_donante, (list, np.ndarray)): anios_donante = [anios_donante]
    if not isinstance(anios_receptor, (list, np.ndarray)): anios_receptor = [anios_receptor]

    # 1. Identify the "deck" of donors
    # pandas isna checks for NaN/None
    mask_donantes = np.isin(year, anios_donante) & ~pd.isna(valores)
    mazo_donantes = valores[mask_donantes]
    
    # 2. Safety check
    if len(mazo_donantes) == 0:
        return valores
    
    # 3. Identify indices to impute
    mask_receptor = pd.isna(valores) & np.isin(year, anios_receptor)
    indices_a_imputar = np.where(mask_receptor)[0]
    
    # 4. Replace NAs
    if len(indices_a_imputar) > 0:
        valores_imputados = np.random.choice(mazo_donantes, size=len(indices_a_imputar), replace=True)
        valores[indices_a_imputar] = valores_imputados
        
    return valores

def imputar_regresion(valores, anios):
    """
    Imputation by Temporal Linear Regression per Group.
    """
    valores = np.array(valores, dtype=float)
    anios = np.array(anios).reshape(-1, 1)
    
    df_temporal = pd.DataFrame({'estimacion': valores, 'anio': anios.flatten()})
    
    entrenamiento = df_temporal.dropna()
    a_predecir = df_temporal[df_temporal['estimacion'].isna()]
    
    if len(entrenamiento) < 2 or len(a_predecir) == 0:
        return valores
    
    X_train = entrenamiento[['anio']]
    y_train = entrenamiento['estimacion']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_pred = a_predecir[['anio']]
    valores_predichos = model.predict(X_pred)
    
    valores[pd.isna(valores)] = valores_predichos
    return valores

def imputar_regresion_filtrada(valores, anios, anios_modelo):
    """
    Imputation by Linear Regression using specific years for the model.
    """
    valores = np.array(valores, dtype=float)
    anios = np.array(anios).reshape(-1, 1)
    
    df_temporal = pd.DataFrame({'valor': valores, 'anio': anios.flatten()})
    
    entrenamiento = df_temporal[~df_temporal['valor'].isna() & df_temporal['anio'].isin(anios_modelo)]
    a_predecir = df_temporal[df_temporal['valor'].isna()]
    
    if len(entrenamiento) < 2 or len(a_predecir) == 0:
        return valores
    
    X_train = entrenamiento[['anio']]
    y_train = entrenamiento['valor']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_pred = a_predecir[['anio']]
    valores_predichos = model.predict(X_pred)
    
    valores[pd.isna(valores)] = valores_predichos
    return valores

def imputar_hibrido_regresion(valores, anios):
    """
    Hybrid Imputation: Regression with Fallback to Mean/Median.
    """
    valores = np.array(valores, dtype=float)
    anios = np.array(anios).reshape(-1, 1)
    
    df_temporal = pd.DataFrame({'estimacion': valores, 'anio': anios.flatten()})
    
    entrenamiento = df_temporal.dropna()
    a_predecir = df_temporal[df_temporal['estimacion'].isna()]
    
    if len(a_predecir) == 0:
        return valores
    
    if len(entrenamiento) >= 2:
        X_train = entrenamiento[['anio']]
        y_train = entrenamiento['estimacion']
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        X_pred = a_predecir[['anio']]
        valores_predichos = model.predict(X_pred)
        valores[pd.isna(valores)] = valores_predichos
        
    elif len(entrenamiento) > 0:
        valor_de_relleno = entrenamiento['estimacion'].median()
        valores[pd.isna(valores)] = valor_de_relleno
        
    return valores

def print_boxplots(input_data, ruta_completa_pdf):
    """
    Function to create a PDF with the graphical representation of the boxplots.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    columnas_numericas = input_data.select_dtypes(include=[np.number]).columns
    
    with PdfPages(ruta_completa_pdf) as pdf:
        for nombre_columna in columnas_numericas:
            plt.figure(figsize=(8, 6))
            try:
                sns.boxplot(y=input_data[nombre_columna], color="lightseagreen", 
                            flierprops=dict(markerfacecolor='red', marker='D', markersize=5))
                plt.title(f"Boxplot of {nombre_columna}")
                plt.ylabel(nombre_columna)
                plt.tight_layout()
                pdf.savefig()
            except Exception as e:
                print(f"Error processing {nombre_columna}: {e}")
            plt.close()

def linear_regression_assumptions(model, X, y):
    """
    Diagnostic checks for linear regression.
    Assumes 'model' is a statsmodels OLS ResultsWrapper.
    """
    
    print("\n========================================================")
    print("       START OF MODEL DIAGNOSTICS")
    print("========================================================\n")
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    # 1. Normality (Shapiro-Wilk)
    print("--- (2) Shapiro-Wilk Test for Normality of Residuals ---")
    shapiro_test = stats.shapiro(residuals)
    print(shapiro_test)
    print("\n--------------------------------------------------------")
    
    # 2. Zero Mean (T-Test)
    print("--- (3) T-Test for a Mean of Residuals equal to Zero ---")
    t_test = stats.ttest_1samp(residuals, 0)
    print(t_test)
    print("\n--------------------------------------------------------")
    
    # 3. Homoscedasticity (Breusch-Pagan)
    print("--- (4) Breusch-Pagan Test for Homoscedasticity ---")
    try:
        bp_test = het_breuschpagan(residuals, model.model.exog)
        print(f"Lagrange multiplier statistic: {bp_test[0]}")
        print(f"p-value: {bp_test[1]}")
        print(f"f-value: {bp_test[2]}")
        print(f"f p-value: {bp_test[3]}")
    except:
        print("Could not run BP test (exog matrix missing)")
        
    print("\n--------------------------------------------------------")
    
    # 4. Independence (Durbin-Watson)
    print("--- (5) Durbin-Watson Test for Autocorrelation ---")
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat}")
    
    print("\n========================================================")
    print("        END OF MODEL DIAGNOSTICS")
    print("========================================================\n")
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Residuals vs Fitted
    sns.residplot(x=fitted, y=residuals, lowess=True, ax=axes[0, 0], 
                  line_kws={'color': 'red', 'lw': 1})
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    
    # QQ Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')
    
    # Scale-Location
    standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
    axes[1, 0].scatter(fitted, standardized_residuals, alpha=0.5)
    sns.regplot(x=fitted, y=standardized_residuals, scatter=False, ax=axes[1, 0], ci=False, lowess=True,
                line_kws={'color': 'red', 'lw': 1})
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)')
    
    # Residuals vs Index
    axes[1, 1].plot(residuals)
    axes[1, 1].set_title('Residuals vs Index')
    
    plt.tight_layout()
    # plt.show() # Commented out to avoid blocking execution in non-interactive envs

def plot_confidence_intervals(result, dataset, x_col="PC1", y_col=None):
    """
    Draw confidence and prediction intervals for a statsmodels result.
    dataset should contain the columns used in regression.
    """
    
    if y_col is None:
        y_col = dataset.columns[0] 
        
    predictions = result.get_prediction(dataset[[x_col ]]) 
    
    summary_frame = predictions.summary_frame(alpha=0.05)
    
    plot_data = pd.concat([dataset.reset_index(drop=True), summary_frame.reset_index(drop=True)], axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.fill_between(plot_data[x_col], plot_data['obs_ci_lower'], plot_data['obs_ci_upper'], 
                     color='green', alpha=0.2, label='Prediction Interval (95%)')
    
    plt.fill_between(plot_data[x_col], plot_data['mean_ci_lower'], plot_data['mean_ci_upper'], 
                     color='red', alpha=0.4, label='Confidence Interval (95%)')
    
    plt.plot(plot_data[x_col], plot_data['mean'], color='darkred', linewidth=2, label='Fit')
    
    plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, color='steelblue', label='Data')
    
    plt.title("Regression with Confidence and Prediction Intervals")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    # plt.show() # Commented out
