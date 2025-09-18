#Year-by-year analysis. I create a model with 2016 and predict 2017. Adjust. Predict 2018. Adjust. Until 2023.
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


# Assumption: 'ev_clean_02' is your complete, clean dataframe containing the 'year' column.

# --- STEP 1: Define the Window Size ---
# The model will learn from the last 'window_size' years of data.
# You can experiment by changing this value (e.g., 2, 3, 4).
# --- Define Parameters ---
# A window_size > 1 is recommended for model stability
window_size <- 1 
# Dynamically set prediction years based on the data range
start_year <- min(ev_clean_02$year)
end_year <- max(ev_clean_02$year)
prediction_years <- (start_year + window_size):end_year

# Initialize vectors to store the results
sliding_rmse_results <- c()
baseline_rmse_results <- c()

# --- Main Sliding Window Loop ---
for (current_year in prediction_years) {
  
  # --- A. Data Splitting ---
  cat(paste("\n--- Processing prediction for year:", current_year, "---\n"))
  
  # Define the time window for the training set
  training_years <- (current_year - window_size):(current_year - 1)
  
  # Filter the main dataframe to create training and test sets
  training_data <- ev_clean_02 %>% filter(year %in% training_years)
  test_data     <- ev_clean_02 %>% filter(year == current_year)
  
  cat(paste("Training with", nrow(training_data), "observations from years:",
            min(training_years), "to", max(training_years), "\n"))
  cat(paste("Testing on", nrow(test_data), "observations from year:", current_year, "\n"))
  
  # --- B. Prepare Predictors (X) and Target (y) ---
  
  # 1. Select predictor variables (all numeric columns except identifiers and the target)
  x_train <- training_data %>% select(where(is.numeric), -year, -EV.Share....)
  x_test  <- test_data %>% select(where(is.numeric), -year, -EV.Share....)
  
  # 2. Log-transform the target variable for the training set
  # We use log1p(x) which calculates log(1+x) to handle potential zero values gracefully.
  y_train_log <- log1p(training_data$EV.Share....)
  
  # 3. Keep the original, non-transformed test target for final evaluation
  y_test_original <- test_data$EV.Share....
  
  # --- C. Calculate Baseline RMSE (on the original scale) ---
  
  # The baseline prediction is the mean of the log-transformed training data
  log_mean_prediction <- mean(y_train_log)
  
  # Back-transform the baseline prediction to the original scale using the exponential function
  # expm1(x) calculates exp(x) - 1, which is the inverse of log1p(x)
  baseline_prediction_original_scale <- expm1(log_mean_prediction)
  
  # Calculate RMSE by comparing the back-transformed baseline against the original test values
  baseline_rmse_value <- sqrt(mean((y_test_original - baseline_prediction_original_scale)^2))
  baseline_rmse_results <- c(baseline_rmse_results, baseline_rmse_value)
  
  # --- D. Train the PCA + Linear Regression Model (on the log scale) ---
  
  pca_map <- prcomp(x_train, scale. = TRUE)
  # Select the first 3 principal components as predictors
  train_pca_scores <- as.data.frame(pca_map$x[, 1:3])
  
  # Create a new dataframe for training the linear model with the log-transformed target
  training_df_for_lm <- data.frame(Log_EV_Share = y_train_log, train_pca_scores)
  
  # The linear model learns to predict the LOG of EV.Share
  lm_model <- lm(Log_EV_Share ~ PC1 + PC2 + PC3, data = training_df_for_lm)
  
  # --- E. Generate and Back-Transform Predictions ---
  
  # Project the test data onto the principal components
  test_pca_scores <- predict(pca_map, newdata = x_test)[, 1:3]
  test_pca_scores_df <- as.data.frame(test_pca_scores)
  
  # The model's initial predictions are on the log scale
  log_predictions <- predict(lm_model, newdata = test_pca_scores_df)
  
  # Back-transform the predictions to the original scale
  final_predictions <- expm1(log_predictions)
  
  # --- F. Evaluate and Store the Model's Result (on the original scale) ---
  
  model_rmse_value <- sqrt(mean((y_test_original - final_predictions)^2))
  sliding_rmse_results <- c(sliding_rmse_results, model_rmse_value)
  
  # Print the results for the current year
  cat(paste("  -> Baseline RMSE for", current_year, ":", round(baseline_rmse_value, 5), "\n"))
  cat(paste("  -> Model RMSE for   ", current_year, ":", round(model_rmse_value, 5), "\n"))
}

# ===================================================================
# 3. ANALYZE FINAL RESULTS
# ===================================================================

# Create a final dataframe with the results from all years
final_results_df <- data.frame(
  Predicted_Year  = prediction_years,
  Model_RMSE      = sliding_rmse_results,
  Baseline_RMSE   = baseline_rmse_results
)

# Calculate the average performance across all folds
average_model_rmse <- mean(sliding_rmse_results)
average_baseline_rmse <- mean(baseline_rmse_results)

# Print a summary of the final results
cat("\n=========================================================\n")
cat("            Sliding Window - Final Results             \n")
cat("=========================================================\n\n")

print(final_results_df)

cat(paste("\nAverage Model Performance (Mean RMSE):   ", round(average_model_rmse, 5), "\n"))
cat(paste("Average Baseline Performance (Mean RMSE):", round(average_baseline_rmse, 5), "\n"))
cat("=========================================================\n")