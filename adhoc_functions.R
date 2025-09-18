#This code will include some functions to be called later

# Calculate the mode of a dataset

getmode <- function(v, na.rm = FALSE) {
  
  # Step 1: Check if the user wants to remove NA values.
  # If na.rm is TRUE, the code inside the if statement is executed.
  if (na.rm) {
    # Filters the vector, keeping only the elements that are NOT NA.
    v <- v[!is.na(v)]
  }
  
  # Step 2 (Optional but recommended): Also remove empty text strings ("").
  v <- v[nchar(as.character(v)) > 0]
  
  # Step 3: Safety check.
  # If the vector is empty after cleaning, there is no mode to calculate.
  if (length(v) == 0) {
    # We return NA to indicate that the mode could not be calculated.
    # NA_character_ is a specific type of NA for text vectors.
    return(NA_character_) 
  }
  
  # Step 4: Calculate the mode (same logic as before).
  # This is executed on the already cleaned vector 'v' (if na.rm = TRUE).
  uniqv <- unique(v)
  return(uniqv[which.max(tabulate(match(v, uniqv)))])
}

#Function to create a PDF with the graphical representation of the different variables of a dataset. 

print_atributes <- function(input_data){
  
  # Loading the libraries is a good practice
  library(ggplot2)
  library(dplyr)
  
  # Opens a PDF device to save the plots
  pdf("E:/Oscar/DataScience/Kryterion/Trabajo final/Variable_Analysis.pdf", width = 10, height = 7)
  
  # 3. Define a threshold to decide if a numeric variable is discrete or continuous
  # If a numeric variable has LESS than this number of unique values, we will make a bar chart.
  # If it has more, we will make a histogram. You can adjust this value.
  umbral_valores_unicos <- 5

  
  # 4. Loop to iterate through each column and generate a plot
  # Iterates through the column names of your data frame
  for (nombre_columna in names(input_data)) {

    # We extract only the column we are interested in and put it in a simple df.
    # The column within this temporary df will be called 'datos_columna'.
    plot_df <- data.frame(datos_columna = input_data[[nombre_columna]])
 
    # Now we build the ggplot using this temporary df and a fixed column name ('datos_columna')
    p <- ggplot(plot_df, aes(x = datos_columna)) +
      labs(title = paste("Distribution of", nombre_columna), x = nombre_columna, y = "Frequency") +
      theme_minimal()
    
    #print(nombre_columna)
    # --- Logic to decide the type of plot ---
    
    # If the column is of type character or factor...

    if (is.character(plot_df[,1]) || is.factor(plot_df[,1])) {
      
      # ...we make a bar chart.
      p <- p + geom_bar(fill = "steelblue", alpha = 0.8)
      print('debugger1')
      # If the column is of type numeric...
    } else if (is.numeric(plot_df[,1])) {
      
      # ...we check how many unique values it has (ignoring NAs).
      n_unicos <- length(unique(na.omit(plot_df[,1])))

      print('debugger2')
      # If it has fewer unique values than our threshold...
      if (n_unicos < umbral_valores_unicos) {
        # ...we treat it as categorical and make a bar chart.
        # It's important to convert it to a factor so ggplot treats it as such.
        p <- ggplot(plot_df, aes(x = factor(.data))) +
          labs(title = paste("Distribution of", nombre_columna), x = nombre_columna, y = "Frequency") +
          theme_minimal() +
          geom_bar(fill = "skyblue", alpha = 0.8)
        print('debugger3')
        # If it has many unique values...
      } else {
        # ...we treat it as continuous and make a histogram.
        p <- p + geom_histogram(bins = 30, fill = "salmon", alpha = 0.8, color="white")
        print('debugger4')
      }
      
    } else {
      # If the column is neither numeric nor text/factor, we skip it.
      next
    }
    
    # We print the plot. Inside a loop, it is necessary to use print().
    print(p)

  }
  # Closes the PDF device to finalize and save the file
  dev.off()
}


# Function to perform time-driven hot-deck imputation
# Allows introducing as a vector both the years that will be donors and the years that will be recipients
imputar_con_tendencia <- function(valores, year, años_donante, años_receptor) {
  
  # 1. Identify the "deck" of donors: values from 2018 and 2019 that are not NA
  mazo_donantes <- valores[year %in% años_donante & !is.na(valores)]
  
  # 2. Safety check: if there are no donors for this state, we do nothing
  if (length(mazo_donantes) == 0) {
    # We return the original values unchanged for this group
    return(valores)
  }
  
  # 3. Identify the indices of the values to be imputed (recipients)
  indices_a_imputar <- which(is.na(valores) & year %in% años_receptor)
  
  # 4. If there are values to impute, we replace them
  if (length(indices_a_imputar) > 0) {
    # For each NA, we draw a random sample (with replacement) from the donor deck
    valores_imputados <- sample(mazo_donantes, size = length(indices_a_imputar), replace = TRUE)
    
    # We replace the NAs with the new values
    valores[indices_a_imputar] <- valores_imputados
  }
  
  # 5. Return the complete vector with the values already imputed
  return(valores)
}
  
  #' Imputation by Temporal Linear Regression per Group
  #'
  #' Imputes NA values in a numeric vector ('prices') by fitting a
  #' linear regression model (prices ~ years) with the observed data and
  #' predicting the values for the NAs.
  #'
  #' @param precios Numeric vector with the prices (can contain NAs).
  #' @param anios Numeric vector with the years corresponding to each price.
  #'
  #' @return A numeric vector with the NAs imputed.
  
imputar_regresion <- function(valores, anios) {
    
    # 1. Create a temporary data frame to work more easily
    df_temporal <- data.frame(estimacion = valores, anio = anios)
    
    # 2. Separate the data into:
    #    - 'training': rows with observed prices (to build the model)
    #    - 'to_predict': rows with missing prices (to use the model)
    entrenamiento <- df_temporal[!is.na(df_temporal$estimacion), ]
    a_predecir <- df_temporal[is.na(df_temporal$estimacion), ]
    
    # 3. Safety check: if there is no data to train or nothing to predict,
    #    or if there is not enough data for a regression (we need at least 2 points),
    #    we return the original data unchanged for this group.
    if (nrow(entrenamiento) < 2 || nrow(a_predecir) == 0) {
      return(valores)
    }
    
    # 4. Fit the linear regression model: price as a function of year
    modelo_lineal <- lm(estimacion ~ anio, data = entrenamiento)
    
    # 5. Predict the prices for the missing years
    valores_predichos <- predict(modelo_lineal, newdata = a_predecir)
    
    # 6. Fill the gaps (NAs) in the original vector with the predictions
    valores[is.na(valores)] <- valores_predichos
    
    # 7. Return the complete and already imputed vector
    return(valores)
}


#Improved and more flexible version of the function to impute by regression
#' Imputation by Linear Regression using specific years for the model
#'
#' @param valores Numeric vector with the data to be imputed (contains NAs).
#' @param anios Numeric vector with the years of each observation.
#' @param anios_modelo Numeric vector with the years that will be used to train the regression model.

imputar_regresion_filtrada <- function(valores, anios, anios_modelo) {
  
  # 1. Create a temporary data frame
  df_temporal <- data.frame(valor = valores, anio = anios)
  
  # 2. Define the training set using ONLY the specified years
  entrenamiento <- df_temporal[!is.na(df_temporal$valor) & df_temporal$anio %in% anios_modelo, ]
  
  # 3. Identify the rows that need to be predicted (all those with NA)
  a_predecir <- df_temporal[is.na(df_temporal$valor), ]
  
  # 4. Safety check
  if (nrow(entrenamiento) < 2 || nrow(a_predecir) == 0) {
    return(valores)
  }
  
  # 5. Fit the linear regression model
  modelo_lineal <- lm(valor ~ anio, data = entrenamiento)
  
  # 6. Predict the prices for the missing years
  valores_predichos <- predict(modelo_lineal, newdata = a_predecir)
  
  # 7. Fill the NAs
  valores[is.na(valores)] <- valores_predichos
  
  # 8. Return the imputed vector
  return(valores)
}

#' Hybrid Imputation: Regression with Fallback to Mean/Median
#'
#' Imputes NA values. If there is enough data (>=2), it uses linear regression (value ~ year).
#' If there is little data (>0 but <2), it uses the median of the available values as a fallback.
#'
#' @param valores Numeric vector with the data to be imputed.
#' @param anios Numeric vector with the corresponding years.
#'
#' @return A numeric vector with the NAs imputed.

imputar_hibrido_regresion <- function(valores, anios) {
  
  # 1. Create a temporary data frame
  df_temporal <- data.frame(estimacion = valores, anio = anios)
  
  # 2. Separate training data and data to be predicted
  entrenamiento <- df_temporal[!is.na(df_temporal$estimacion), ]
  a_predecir <- df_temporal[is.na(df_temporal$estimacion), ]
  
  # 3. Initial check: if there is nothing to impute, we exit.
  if (nrow(a_predecir) == 0) {
    return(valores)
  }
  
  # --- IMPROVED HYBRID LOGIC ---
  
  # CASE 1: There is enough data for a regression
  if (nrow(entrenamiento) >= 2) {
    
    # Message to know which method is being used (useful for debugging)
    # message("Using linear regression...")
    
    modelo_lineal <- lm(estimacion ~ anio, data = entrenamiento)
    valores_predichos <- predict(modelo_lineal, newdata = a_predecir)
    valores[is.na(valores)] <- valores_predichos
    return(valores)
    
    # CASE 2 (FALLBACK): There is some data, but not enough for regression (i.e., only 1)
  } else if (nrow(entrenamiento) > 0) {
    
    # message("Insufficient data for regression. Using fallback to the median.")
    
    # We calculate the median of the few data we have (if there is only one, it will be that same value)
    valor_de_relleno <- median(entrenamiento$estimacion, na.rm = TRUE)
    
    # We fill all NAs with that single value
    valores[is.na(valores)] <- valor_de_relleno
    return(valores)
    
    # CASE 3: There is no training data at all
  } else {
    # We can't do anything, we return the data as it was
    # message("No data available to impute.")
    return(valores)
  }
}

#Function to create a PDF with the graphical representation of the boxplots of the different variables.

print_boxplots <- function(input_data, ruta_completa_pdf) {
  
  # Load necessary libraries
  library(ggplot2)
  library(dplyr)
  
  # 1. Identify only the columns that are numeric
  columnas_numericas <- names(input_data)[sapply(input_data, is.numeric)]
  
  # Informative message
  message(paste("Boxplots will be generated for", length(columnas_numericas), "numeric variables."))
  
  # 2. Open the PDF device to save the plots
  pdf(ruta_completa_pdf, width = 8, height = 6)
  
  # 3. Loop to iterate through each numeric column
  for (nombre_columna in columnas_numericas) {
    
    # Message to see the progress in the console
    message(paste("Creating boxplot for:", nombre_columna))
    
    # We create the plot using tryCatch to prevent an error from stopping everything
    tryCatch({
      
      p <- ggplot(input_data, aes(x = "", y = .data[[nombre_columna]])) + # <- The key is here
        geom_boxplot(
          fill = "lightseagreen", # Fill color
          alpha = 0.7,
          outlier.color = "red",  # Highlight outliers in red
          outlier.shape = 18,     # Change the shape of the outlier point
          outlier.size = 2        # Make outliers a bit larger
        ) +
        labs(
          title = paste("Boxplot of", nombre_columna),
          subtitle = "For outlier detection",
          x = "",                  # We don't need a label on the X-axis
          y = nombre_columna
        ) +
        theme_minimal()
      
      # We print the plot into the PDF
      print(p)
      
    }, error = function(e) {
      # If an error occurs, print a message and continue
      message(paste("   -> ERROR processing '", nombre_columna, "'. Error:", e$message))
    }) # End of tryCatch
    
  } # End of the for loop
  
  # 4. Close the PDF device to finalize and save the file
  dev.off()
  
  message(paste("Process finished. The PDF file has been created at:", ruta_completa_pdf))
}

linear_regression_assumptions <- function(regression_model) {
  
  # --- Force the printing of each test ---
  
  cat("\n========================================================\n")
  cat("       START OF MODEL DIAGNOSTICS\n")
  cat("========================================================\n\n")
  
  # 1. Specification (Ramsey RESET Test)
  cat("--- (1) Ramsey RESET Test for Model Specification ---\n\n")
  print(resettest(regression_model))
  cat("\n--------------------------------------------------------\n")
  
  # 2. Normality of Residuals (Shapiro-Wilk Test)
  cat("--- (2) Shapiro-Wilk Test for Normality of Residuals ---\n\n")
  print(shapiro.test(regression_model$residuals))
  cat("\n--------------------------------------------------------\n")
  
  # 3. Zero Mean of Errors (T-Test)
  cat("--- (3) T-Test for a Mean of Residuals equal to Zero ---\n\n")
  print(t.test(regression_model$residuals, mu = 0))
  cat("\n--------------------------------------------------------\n")
  
  # 4. Homoscedasticity (Breusch-Pagan Test)
  cat("--- (4) Breusch-Pagan Test for Homoscedasticity ---\n\n")
  print(bptest(regression_model))
  cat("\n--------------------------------------------------------\n")
  
  # 5. Independence of Errors (Durbin-Watson Test)
  cat("--- (5) Durbin-Watson Test for Autocorrelation ---\n\n")
  print(durbinWatsonTest(regression_model))
  
  cat("\n========================================================\n")
  cat("        END OF MODEL DIAGNOSTICS\n")
  cat("========================================================\n\n")
  
  # --- Generate the plots ---
  # RStudio will show them in the "Plots" tab
  
  par(mfrow = c(2, 2)) # Prepares a 2x2 grid for the plots
  plot(regression_model)
  par(mfrow = c(1, 1)) # Restores the configuration to a single plot
}

plot_confidence_intervals <- function(regression_model,dataset){
  #Draw confidence and prediction intervals.
  # 2. Calculate both intervals (confidence and prediction) at 95%
  intervalo_confianza <- predict(regression_model, interval = "confidence", level = 0.95)
  colnames(intervalo_confianza)[1:3] <- c("fit","conf_lwr","conf_upr")
  
  intervalo_prediccion <- predict(regression_model, interval = "prediction", level = 0.95)
  intervalo_prediccion <- intervalo_prediccion[,-1]
  colnames(intervalo_prediccion)[1:2] <- c("pred_lwr","pred_upr")
  
  # 3. Add these intervals to our original data frame
  total_data_con_intervalos <- cbind(dataset, intervalo_confianza, intervalo_prediccion)
  
  
  # We see how the data frame looks
  head(total_data_con_intervalos)
  
  #Create the plot with ggplot
  ggplot(total_data_con_intervalos, aes(x = PC1, y = dataset[,1])) +
    
    # Layer 1: Prediction Interval (the wider band)
    geom_ribbon(aes(ymin = pred_lwr, ymax = pred_upr), 
                fill = "green", 
                alpha = 0.2) +
    
    # Layer 2: Confidence Interval (the narrower band)
    geom_ribbon(aes(ymin = conf_lwr, ymax = conf_upr), 
                fill = "red", 
                alpha = 0.4) +
    
    # Layer 3: The regression line
    geom_line(aes(y = fit), color = "darkred", size = 1) +
    
    # Layer 4: The scatter plot
    geom_point(alpha = 0.6, color = "steelblue") +
    
    labs(
      title = "Regression with Confidence and Prediction Intervals",
      subtitle = "Wide band = Prediction (95%), Narrow band = Confidence (95%)",
      x = "Principal Component 1 (PC1)",
      y = "Target Variable"
    ) +
    theme_minimal()
}