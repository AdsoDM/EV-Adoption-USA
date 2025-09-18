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

#Load the complete and imputed datasets


ev_clean_01 <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/ev_adoption_clean.csv")


#Encoding of the Party variable. I assign value 1 to the Democratic party because its policies are usually more aligned with environmentalism and the fight against climate change

ev_clean_01 <- ev_clean_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))


#the new column is of type factor and I have to convert it first to character and then to number because otherwise it gives an error

ev_clean_01 <- ev_clean_01 %>% mutate(Party = as.numeric(as.character(Party)))


#Dimensionality reduction analysis

#scaling of features without the index variable and leaving out the state variable, which is what I will use to analyze the PCA results

#I will select numeric variables for this analysis

ev_clean_02 <- ev_clean_01 %>% select_if(is.numeric)

#I remove the Index variable as it does not provide variance or correlation information since it is a categorical variable (a numbered list)
ev_clean_02 <- ev_clean_02 %>% select(-Index)

#I will filter the data to select 2016 data. I will delete the columns with standar deviation = 0

ev_clean_2016 <- ev_clean_02 %>% filter(year=="2016")
ev_clean_2016 <- ev_clean_2016 %>% select(-year)
ev_clean_2016 <- ev_clean_2016 %>% select(-fuel_economy)


#I will create row names using the  state from the ev_clean_01 dataset

row_names_evclean <- ev_clean_01 %>% filter(year=="2016") %>% select(state)
row.names(ev_clean_2016) <- row_names_evclean$state


#I will separate both sets into dependent and not dependent variables

x_train_2016 <- ev_clean_2016 %>% select(-EV.Share....)
y_train_2016 <- ev_clean_2016 %>% select (EV.Share....)


#I create the map to explore the variables using dimensionality reduction
map_pca_2016 <- prcomp(x_train_2016, scale = TRUE)
summary(map_pca_2016)

train_pca_2016 <- as.data.frame(map_pca_2016$x[, 1:2])


#I create the dataset to train the model with
data_final_2016 <- data.frame(EV.Share_predict = y_train_2016[,1], train_pca_2016)
lm_model_2c_2016 <- lm(EV.Share_predict ~ PC1+PC2, data=data_final_2016)

summary(lm_model_2c_2016)
#linear_regression_assumptions(lm_model_2c)
plot_confidence_intervals(lm_model_2c,data_final_2016)

#Ahora voy a utilizar el comando 'predict' para mapear los valores de 2017 en el PCA de 2016
#I will filter the data to select 2016 data. I will delete the columns with standar deviation = 0

ev_clean_2017 <- ev_clean_02 %>% filter(year=="2017") %>% select(-year,-fuel_economy)

#I will create row names using the  state from the ev_clean_01 dataset

row.names(ev_clean_2017) <- row_names_evclean$state


#I will separate both sets into dependent and not dependent variables

x_test_2017 <- ev_clean_2017 %>% select(-EV.Share....)
y_test_2017 <- ev_clean_2017 %>% select (EV.Share....)

# We trasform 2017 dataset with PCA previoulsy map
test_pca_2017 <- predict(map_pca_2016, newdata = x_train_2017)[, 1:2]

#Conversiont into dataframe
test_pca_2017_df <- as.data.frame(test_pca_2017)

#I use the 2016 model to predict the next year

predictions_2017 <- predict(lm_model_2c_2016, newdata = test_pca_2017_df)

#Evaluation
results <- data.frame(
  Actual_2017 = y_test_2017$EV.Share....,
  Predicted_2017 = predictions_2017
)
print(head(results))

#Calculation of RMSE
rmse <- sqrt(mean((results$Actual_2017 - results$Predicted_2017)^2))
print(paste("The prediction error (RMSE) of the 2016 model with the new data is:", rmse))

# Calcula la desviación estándar de la variable objetivo de 2017
sd_2017 <- sd(y_test_2017$EV.Share....)
print(paste("La desviación estándar de EV.Share en 2017 es:", sd_2017))

# 1. Calcula la media de EV.Share en los datos de entrenamiento (2016)
mean_2016 <- mean(y_train_2016$EV.Share....)

# 2. Calcula el error de este modelo base en los datos de 2017
rmse_baseline <- sqrt(mean((y_test_2017$EV.Share.... - mean_2016)^2))

print(paste("El RMSE del modelo base (predecir la media) es:", rmse_baseline))
