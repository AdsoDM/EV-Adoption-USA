#Year-by-year analysis. I create a model with 2016 and predict 2017. Adjust. Predict 2018. Adjust. Until 2023.

#Añado el 'Año' como variable predictora en la regresión

#load libraries

library(lmtest) #checking normality of errors in a regression
library(tibble) #To convert a column into row names
library(ggplot2)
library(ggfortify) #visualization for PCA
library(gridExtra) #visualization for PCA
library(factoextra) #visualization for PCA. Contains fviz
library(corrplot) #analysis prior to PCA
library(fastDummies) #Necessary for one-hot encoding
library(magrittr) #to use the %>% pipeline
library(dplyr) #to use mutate
library(lmtest) #checking normality of errors in a regression
library(tibble) #To convert a column into row names
library(car) #to perform the Durbin Watson test

#Load my functions
source("E:/Oscar/DataScience/Kryterion/Trabajo final/adhoc_functions.R")

# --- Initialization ---

# This vector will store the Root Mean Squared Error (RMSE) for each iteration.
rmse_results <- c()

#Load the complete and imputed datasets
ev_clean_01 <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/ev_adoption_clean.csv")

#Encoding of the Party variable. I assign value 1 to the Democratic party because its policies are usually more aligned with environmentalism and the fight against climate change
ev_clean_01 <- ev_clean_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))
#the new column is of type factor and I have to convert it first to character and then to number because otherwise it gives an error
ev_clean_01 <- ev_clean_01 %>% mutate(Party = as.numeric(as.character(Party)))

#Expansion Window Analysis

#scaling of features without the index variable and leaving out the state variable, which is what I will use to analyze the PCA results
#I will select numeric variables for this analysis
ev_clean_02 <- ev_clean_01 %>% select_if(is.numeric)
row_names <- paste(ev_clean_01$state,ev_clean_01$year, sep="_")
row.names(ev_clean_02) <- row_names

#I remove the Index and fuel economy variable as it does not provide variance or correlation information since it is a categorical variable (a numbered list) or a 0 variance variable like fuel_economy (in a yearly basis)
ev_clean_02 <- ev_clean_02 %>% select(-Index,-fuel_economy)


# Define the range of years we want to generate predictions for.
# We start with 2017 because we need at least 2016 as our first training set.
prediction_years <- 2017:2022

# --- Main Loop ---

for (current_year in prediction_years) {
  
  # --- A. Data Splitting (The window expands here) ---
  
  cat(paste("\n--- Processing prediction for year:", current_year, "---\n"))
  
  # Define the training years as all years from the beginning up to the year
  # *before* the current prediction year.
  training_years <- 2016:(current_year - 1)
  
  training_data <- ev_clean_02 %>% filter(year %in% training_years)
  test_data     <- ev_clean_02 %>% filter(year == current_year)
  
  cat(paste("Training with", nrow(training_data), "observations from years:",
            min(training_years), "to", max(training_years), "\n"))
  cat(paste("Testing on", nrow(test_data), "observations from year:", current_year, "\n"))
  
  # --- B. Prepare Training Data (Features and Target) ---
  
  x_train <- training_data %>% select(-EV.Share...., -year)
  y_train <- training_data %>% select(EV.Share....)
  
  # --- C. Train the Models (PCA and Linear Regression) ---
  
  # The PCA map is created INSIDE the loop, using only the current training data.
  # This ensures that no information from the test set "leaks" into the model.
  pca_map <- prcomp(x_train, scale = TRUE)
  
  # Extract the principal component scores. We'll use the first 3 components.
  # This number is a hyperparameter you can tune.
  train_pca_scores <- as.data.frame(pca_map$x[, 1:3])
  
  # Combine the PCA scores with the target variable to prepare for regression.
  training_df_for_lm <- data.frame(EV_Share = y_train$EV.Share...., Year = training_data$year, train_pca_scores)
  
  # Train the linear regression model.
  lm_model <- lm(EV_Share ~ PC1 + PC2 + PC3 + Year, data = training_df_for_lm)
  
  # --- D. Prepare Test Data (Features and Target) ---
  
  x_test <- test_data %>% select(-EV.Share...., -year)
  y_test <- test_data %>% select(EV.Share....)
  
  # --- E. Generate Predictions on the Test Set ---
  
  # 1. Transform the test data using the PCA map trained on the training data.
  test_pca_scores <- predict(pca_map, newdata = x_test)[, 1:3]
  test_pca_scores_df <- as.data.frame(test_pca_scores)
  test_pca_scores_df$Year <- current_year
  
  # 2. Use the trained linear model to make predictions.
  predictions <- predict(lm_model, newdata = test_pca_scores_df)
  
  # --- F. Evaluate and Store the Result ---
  
  # Calculate the RMSE for the current year's predictions.
  rmse_value <- sqrt(mean((y_test$EV.Share.... - predictions)^2))
  
  cat(paste("RMSE for", current_year, ":", round(rmse_value, 5), "\n"))
  
  # Append the RMSE to our results vector.
  rmse_results <- c(rmse_results, rmse_value)
}

# ===================================================================
# Step 3: Analyze Final Results
# ===================================================================

# Create a final dataframe to display the results clearly.
final_results_df <- data.frame(
  Predicted_Year = prediction_years,
  RMSE = rmse_results
)

# Calculate the average performance of the model across all years.
average_rmse <- mean(rmse_results)

cat("\n=========================================================\n")
cat("          Expanding Window Cross-Validation Results        \n")
cat("=========================================================\n\n")

print(final_results_df)

cat(paste("\nAverage Model Performance (Mean RMSE):", round(average_rmse, 5), "\n"))
cat("=========================================================\n")
