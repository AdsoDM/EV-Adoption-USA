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
library(tidyr)
library(lmtest) #checking normality of errors in a regression
library(tibble) #To convert a column into row names
library(car) #to perform the Durbin Watson test
library(caret) # to use the nearzerovar() function

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
# Remove columns that are not useful for the YoY analysis
ev_clean_02 <- ev_clean_01 %>% 
  select(-Index, -fuel_economy)

# ===================================================================
# 2. YEAR-OVER-YEAR (YoY) TRANSFORMATION
# ===================================================================
yoy_data <- ev_clean_02 %>%
  group_by(state) %>%
  arrange(year, .by_group = TRUE) %>%
  mutate(across(
    .cols = where(is.numeric) & !year,
    .fns = ~ (. - lag(.)) / (lag(.) + 1e-9),
    .names = "{.col}_YoY_Pct_Change"
  )) %>%
  ungroup() %>%
  na.omit()

# ===================================================================
# 3. MODELING DATA PREPARATION
# ===================================================================
model_data_yoy <- yoy_data %>%
  select(year, state, ends_with("_YoY_Pct_Change")) %>%
  unite(col = "observation_id", state, year, sep = "_", remove = FALSE)

# Robustly identify the target variable name ONCE
target_variable_name <- grep("^EV\\.Share.*_YoY_Pct_Change$", names(model_data_yoy), value = TRUE)
cat("Identified Target Variable:", target_variable_name, "\n")

# ===================================================================
# 4. SLIDING WINDOW CROSS-VALIDATION
# ===================================================================

# --- Define Parameters ---
window_size <- 3
prediction_years <- (min(model_data_yoy$year) + window_size):max(model_data_yoy$year)

# Initialize vectors to store results
sliding_rmse_results <- c()
baseline_rmse_results <- c()

# --- Main Loop ---
for (current_year in prediction_years) {
  
  cat(paste("\n--- Processing prediction for year:", current_year, "---\n"))
  
  # A. Define time windows and split data
  training_years <- (current_year - window_size):(current_year - 1)
  training_data <- model_data_yoy %>% filter(year %in% training_years)
  test_data     <- model_data_yoy %>% filter(year == current_year)
  
  cat(paste("Training with", nrow(training_data), "observations...\n"))
  cat(paste("Testing on", nrow(test_data), "observations...\n"))
  
  # B. Prepare initial training and test sets
  x_train_initial <- training_data %>% select(ends_with("_YoY_Pct_Change"), -all_of(target_variable_name))
  y_train <- training_data %>% pull(all_of(target_variable_name))
  
  x_test_initial  <- test_data %>% select(ends_with("_YoY_Pct_Change"), -all_of(target_variable_name))
  y_test  <- test_data %>% pull(all_of(target_variable_name))
  
  # C. Remove zero-variance predictors found in the training set
  zero_variance_predictors <- nearZeroVar(x_train_initial, saveMetrics = FALSE)
  
  if (length(zero_variance_predictors) > 0) {
    cat(paste("Removing", length(zero_variance_predictors), "zero-variance predictor(s).\n"))
    x_train <- x_train_initial[, -zero_variance_predictors]
    x_test  <- x_test_initial[, -zero_variance_predictors]
  } else {
    x_train <- x_train_initial
    x_test  <- x_test_initial
  }
  
  # D. Calculate Baseline RMSE for the current fold
  mean_of_training_data <- mean(y_train)
  baseline_rmse_value <- sqrt(mean((y_test - mean_of_training_data)^2))
  baseline_rmse_results <- c(baseline_rmse_results, baseline_rmse_value)
  
  # E. Train PCA + Regression Model
  pca_map <- prcomp(x_train, scale. = TRUE)
  
  # Select the first 3 principal components
  train_pca_scores <- as.data.frame(pca_map$x[, 1:3])
  
  # Combine target and PCA scores for the linear model
  training_df_for_lm <- data.frame(Target = y_train, train_pca_scores)
  
  lm_model <- lm(Target ~ PC1 + PC2 + PC3, data = training_df_for_lm)
  
  # F. Generate Predictions
  test_pca_scores <- predict(pca_map, newdata = x_test)[, 1:3]
  test_pca_scores_df <- as.data.frame(test_pca_scores)
  
  predictions <- predict(lm_model, newdata = test_pca_scores_df)
  
  # G. Evaluate and Store Model's Result
  model_rmse_value <- sqrt(mean((y_test - predictions)^2))
  sliding_rmse_results <- c(sliding_rmse_results, model_rmse_value)
  
  cat(paste("  -> Baseline RMSE for", current_year, ":", round(baseline_rmse_value, 5), "\n"))
  cat(paste("  -> Model RMSE for   ", current_year, ":", round(model_rmse_value, 5), "\n"))
}

# ===================================================================
# 5. ANALYZE FINAL RESULTS
# ===================================================================
final_results_df <- data.frame(
  Predicted_Year  = prediction_years,
  Model_RMSE      = sliding_rmse_results,
  Baseline_RMSE   = baseline_rmse_results
)

cat("\n=========================================================\n")
cat("            Sliding Window - Final Results             \n")
cat("=========================================================\n\n")
print(final_results_df)
cat(paste("\nAverage Model Performance (Mean RMSE):   ", round(mean(sliding_rmse_results), 5), "\n"))
cat(paste("Average Baseline Performance (Mean RMSE):", round(mean(baseline_rmse_results), 5), "\n"))
cat("=========================================================\n")
