#carga de librerias

library(lmtest) #compraobación de normalidad de los errores de una regresion
library(tibble) #Para convertir una columna en nombres de fila
library(ggplot2)
library(ggfortify)#visulalización para PCA
library(gridExtra)#visulalización para PCA
library(factoextra)#visulalización para PCA. Contiene fviz 
library(corrplot) #análisis previo a PCA
library(fastDummies) #Necesaria para el one hot encoding
library(magrittr) #para poder utilizar el pipeline %>%
library(dplyr) #para poder usar mutate
library(lmtest) #comprobación de normalidad de los errores de una regresion
library(tibble) #Para convertir una columna en nombres de fila
library(car)#para realizar el test de Durbin Watson

#Cargo mis funciones
source("E:/Oscar/DataScience/Kryterion/Trabajo final/adhoc_functions.R")
 
#Cargo los datasets completos e imputados

training_01 <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/training_clean.csv")
test_01 <-  read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/test_clean.csv")

#Codificación de la variable Party. Asigno valor 1 al partido demócrata porque sus políticas sulen ir más alineadas con el ecologismo y la lucha contra el cambio climático

training_01 <- training_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))
test_01 <- test_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))

#la nueva columna es de tipo factor y tengo que pasarla primero a caractery luego a número porque si no da error

training_01 <- training_01 %>% mutate(Party = as.numeric(as.character(Party)))
test_01 <- test_01 %>% mutate(Party = as.numeric(as.character(Party)))

#Análisis de reducción dimensional

#escalado de características sin la variable index y dejando fuera la variable stado, que será lo que voy a utilizar para analizar los resultados del PCA

#Voy a eliminar la columna index que no aporta nada al análisis

training_02 <- training_01 %>% select_if(is.numeric)
test_02 <- test_01 %>% select_if(is.numeric)

#Elimino la variable Index ya que no aporta información de varianza o correlación al tratarse de una variable categórica (una lista numerada)
training_02 <- training_02 %>% select(-Index)
test_02 <- test_02 %>% select(-Index)

#Voy a crear nombres de filas usando el año y el estado del dataset training_01

row_names_training <- paste(training_01$state,training_01$year, sep="_")
row.names(training_02) <- row_names_training
row_names_test <- paste(test_01$state,test_01$year, sep="_")
row.names(test_02) <- row_names_test

#Voy a separar ambos conjuntos en datos pertenecientes a entrenamiento o test

x_train <- training_02 %>% select(-EV.Share....)
y_train <- training_02 %>% select (EV.Share....)

x_test<- test_02 %>% select(-EV.Share....)
y_test <- test_02 %>% select(EV.Share....)


map_pca_01 <- prcomp(x_train, scale = TRUE)
summary(map_pca_01)

#análisis de resultados de pca

fviz_eig(map_pca_01, col.var="blue")
fviz_pca_var(map_pca_01,
             col.var = "cos2",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

#Regresión con dos variables
#Transformamos los conjuntos de datos con el nuevo mapa (PCA) creado con el conjunto de training

train_pca_01 <- predict(map_pca_01, newdata = x_train)[, 1:2]
test_pca_01 <- predict(map_pca_01, newdata = x_test)[, 1:2]

#Creo el dataset con el que entrenar al modelo
train_final_01 <- data.frame(EV.Share_predict = y_train[,1], train_pca_01)
lm_model_2c <- lm(EV.Share_predict ~ PC1+ PC2, data=train_final_01)

summary(lm_model_2c)
linear_regression_assumptions(lm_model_2c)
plot_confidence_intervals(lm_model_2c,train_final_01)


#Regresión con tres variables
#Transformamos los conjuntos de datos con el nuevo mapa (PCA) creado con el conjunto de training

train_pca_02 <- predict(map_pca_01, newdata = x_train)[, 1:3]
test_pca_02 <- predict(map_pca_01, newdata = x_test)[, 1:3]

#Creo el dataset con el que entrenar al modelo
train_final_02 <- data.frame(EV.Share_predict = y_train[,1], train_pca_02)
lm_model_3c <- lm(EV.Share_predict ~ PC1+PC2+PC3, data=train_final_02)

summary(lm_model_3c)
linear_regression_assumptions(lm_model_3c)

plot_confidence_intervals(lm_model_3c,train_final_02)


# Asumimos que 'lm_model_3c' es tu modelo y 'train_final_02' tus datos de entrenamiento
# 1. Obtenemos las predicciones del modelo completo
predicciones <- predict(lm_model_3c, newdata = train_final_02)

# 2. Creamos un data frame para el gráfico
plot_data <- data.frame(
  Reales = train_final_02$EV.Share_predict,
  Predichos = predicciones
)

# 3. Creamos el gráfico
ggplot(plot_data, aes(x = Reales, y = Predichos)) +
  geom_point(alpha = 0.6, color = "blue") +
  # Añadimos la línea de 45 grados que representa una predicción perfecta
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Valores Predichos vs. Reales",
    x = "Valores Reales (Observados)",
    y = "Valores Predichos por el Modelo"
  ) +
  theme_minimal()


############Regresión sobre variable log(EV.Share)

#Voy a separar ambos conjuntos en datos pertenecientes a entrenamiento o test

train_final_03 <- train_final_02
filas_quitar <- c("North Dakota_2016","Mississippi_2016")
train_final_03 <- train_final_03[! rownames(train_final_03) %in% filas_quitar ,]
train_final_03$log_EV.Share <- log(train_final_03[,1])

lm_model_log <- lm(log_EV.Share ~ PC1 + PC2 + PC3, data = train_final_03)
summary(lm_model_log)

linear_regression_assumptions(lm_model_3c)
plot_confidence_intervals(lm_model_3c,train_final_02)
