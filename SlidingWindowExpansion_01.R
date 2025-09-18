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
window_size <- 1

# --- STEP 2: Adjust the Range of Years to Predict ---
# The first year we can predict is 2016 + window_size.
prediction_years <- (2016 + window_size):2022

# Initialize a vector to store the RMSE results
sliding_rmse_results <- c()
baseline_rmse_results <- c() # New vector for baseline results

# --- Main Sliding Window Loop ---

for (current_year in prediction_years) {
  
  # --- A. Data Splitting ---
  cat(paste("\n--- Processing prediction for year:", current_year, "---\n"))
  
  training_years <- (current_year - window_size):(current_year - 1)
  
  training_data <- ev_clean_02 %>% filter(year %in% training_years)
  test_data     <- ev_clean_02 %>% filter(year == current_year)
  
  cat(paste("Training with", nrow(training_data), "observations from years:",
            min(training_years), "to", max(training_years), "\n"))
  cat(paste("Testing on", nrow(test_data), "observations from year:", current_year, "\n"))
  
  # --- B. Prepare Training and Test Data ---
  x_train <- training_data %>% select(-EV.Share...., -year)
  y_train <- training_data %>% select(EV.Share....)
  x_test  <- test_data %>% select(-EV.Share...., -year)
  y_test  <- test_data %>% select(EV.Share....)
  
  # --- C. Calculate Baseline RMSE for the current year ---
  mean_of_training_data <- mean(y_train$EV.Share....)
  baseline_rmse_value <- sqrt(mean((y_test$EV.Share.... - mean_of_training_data)^2))
  
  # Store the baseline result
  baseline_rmse_results <- c(baseline_rmse_results, baseline_rmse_value)
  
  # --- D. Train the PCA + Regression Models ---
  pca_map <- prcomp(x_train, scale = TRUE)
  train_pca_scores <- as.data.frame(pca_map$x[, 1:3])
  training_df_for_lm <- data.frame(EV_Share = y_train$EV.Share...., train_pca_scores)
  lm_model <- lm(EV_Share ~ PC1 + PC2 + PC3, data = training_df_for_lm)
  
  # --- E. Generate Predictions with your model ---
  test_pca_scores <- predict(pca_map, newdata = x_test)[, 1:3]
  test_pca_scores_df <- as.data.frame(test_pca_scores)
  predictions <- predict(lm_model, newdata = test_pca_scores_df)
  
  # --- F. Evaluate and Store the Model's Result ---
  model_rmse_value <- sqrt(mean((y_test$EV.Share.... - predictions)^2))
  sliding_rmse_results <- c(sliding_rmse_results, model_rmse_value)
  
  # Print both results for immediate comparison
  cat(paste("  -> Baseline RMSE for", current_year, ":", round(baseline_rmse_value, 5), "\n"))
  cat(paste("  -> Model RMSE for   ", current_year, ":", round(model_rmse_value, 5), "\n"))
}

# ===================================================================
# Analyze Final Results
# ===================================================================

# Create a final dataframe with both model and baseline results
final_results_df <- data.frame(
  Predicted_Year  = prediction_years,
  Model_RMSE      = sliding_rmse_results,
  Baseline_RMSE   = baseline_rmse_results
)

# Calculate the average performance
average_model_rmse <- mean(sliding_rmse_results)
average_baseline_rmse <- mean(baseline_rmse_results)

cat("\n=========================================================\n")
cat("            Sliding Window - Final Results             \n")
cat("=========================================================\n\n")

print(final_results_df)

cat(paste("\nAverage Model Performance (Mean RMSE):   ", round(average_model_rmse, 5), "\n"))
cat(paste("Average Baseline Performance (Mean RMSE):", round(average_baseline_rmse, 5), "\n"))
cat("=========================================================\n")