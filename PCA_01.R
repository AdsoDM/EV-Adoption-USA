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
 
#Cargo el dataset completo e imputado

ev_clean_01 <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/ev_adoption_clean.csv")

#Codificación de la variable Party. Asigno valor 1 al partido demócrata porque sus políticas sulen ir más alineadas con el ecologismo y la lucha contra el cambio climático

ev_clean_01 <- ev_clean_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))

#la nueva columna es de tipo factor y tengo que pasarla primero a caractery luego a número porque si no da error

ev_clean_01 <- ev_clean_01 %>% mutate(Party = as.numeric(as.character(Party)))


#Análisis de reducción dimensional

#escalado de características sin la variable index y dejando fuera la variable stado, que será lo que voy a utilizar para analizar los resultados del PCA

#Voy a eliminar la columna index que no aporta nada al análisis

ev_clean_02 <- ev_clean_01 %>% select_if(is.numeric)
ev_clean_02 <- ev_clean_02 %>% select(-Index,-year)

#Voy a crear nombres de filas usando el año y el estado del dataset training_01

row_names <- paste(ev_clean_01$state,ev_clean_01$year, sep="_")
row.names(ev_clean_02) <- row_names

ev_clean_pca_01 <- prcomp(ev_clean_02, scale = TRUE)
summary(ev_clean_pca_01)

#análisis de resultados de pca

fviz_eig(ev_clean_pca_01, col.var="blue")
fviz_pca_var(ev_clean_pca_01,
             col.var = "cos2",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

ind <- get_pca_ind(ev_clean_pca_01)
fviz_pca_ind(ev_clean_pca_01,
             col.var = "cos2",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = FALSE)


#Ver la calidad de representación de las distintas observaciones (estado-año)
head(ind$cos2)
#Ver las coordenadas de las primeras filas
head(ind$coord)

#Preparació de las variables para hacer una regresión lineal de los dos primeros componentes

pca_scores <- as.data.frame(ev_clean_pca_01$x[,1:2])
total_data <- data.frame(
  PC1 = pca_scores$PC1,
  PC2 = pca_scores$PC2
)

boxplot(total_data)

#Regresion lineal

ev_clean_pca_R <- lm(PC2 ~ PC1, total_data)
summary(ev_clean_pca_R)

#Explicación de la variable EV_share utilizando PC1 y PC2
total_data_2 <- data.frame(
  PC1 = pca_scores$PC1,
  PC2 = pca_scores$PC2,
  Objective = ev_clean_02$EV.Share....
)

ev_clean_pca_R_02 <- lm(Objective ~ PC1+PC2, total_data_2)
summary(ev_clean_pca_R_02)


#VOy a ver los datos solmente del año 2023 porque si no es dificil observar nada

ev_clean_03 <- ev_clean_01 %>% filter(year==2023)
ev_clean_03 <- ev_clean_03 %>% column_to_rownames(var="state")
ev_clean_03 <- ev_clean_03 %>% select(-Index,-year)
ev_clean_03 <- ev_clean_03 %>% select(-fuel_economy)#elimino la variable fuel economy porque es una variable anual y su valor es constante para el mismo año


ev_clean_pca_02 <- prcomp(ev_clean_03, scale = TRUE)
summary(ev_clean_pca_02)


#análisis de resultados de pca

fviz_eig(ev_clean_pca_02, col.var="blue")
fviz_pca_var(ev_clean_pca_02,
             col.var = "coord",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

ind <- get_pca_ind(ev_clean_pca_02)
fviz_pca_ind(ev_clean_pca_02,
             col.var = "coord",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

#Voy a plotear con diferentes colores los estados republicanos y los estados democratas
fviz_pca_ind(ev_clean_pca_02,
             geom.ind = "point",
             habillage = factor(ev_clean_03$Party,
                                levels = c(0,1),
                                labels = c("Republican","Democrat")),
             addEllipses = TRUE,
             pallete = c("red","blue"),
             legend.tittle = "Political Affiliation"
)+
  labs(tittle = "PCA analisys by Political Affiliation")



#Ver la calidad de representación de las distintas observaciones (estado-año)
head(ind$cos2)
#Ver las coordenadas de las primeras filas
head(ind$coord)

#Antes de preparar la regresión lineal voy a sacar EV_Share del análisis PCA y luego voy a ver como las componentes principales (sin EV_Share) explican EV_share
ev_clean_04 <- ev_clean_03 %>% select(-EV.Share....)
ev_clean_pca_03 <- prcomp(ev_clean_04, scale = TRUE)
summary(ev_clean_pca_03)

fviz_pca_var(ev_clean_pca_03,
             col.var = "coord",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

fviz_pca_ind(ev_clean_pca_03,
             col.var = "coord",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

#Voy a plotear con diferentes colores los estados republicanos y los estados democratas
fviz_pca_ind(ev_clean_pca_03,
             geom.ind = "point",
             habillage = factor(ev_clean_04$Party,
                                levels = c(0,1),
                                labels = c("Republican","Democrat")),
             addEllipses = TRUE,
             pallete = c("red","blue"),
             legend.tittle = "Political Affiliation"
             )+
  labs(tittle = "PCA analisys by Political Affiliation")



#Preparació de las variables para hacer una regresión lineal de los dos primeros componentes

pca_scores_2 <- as.data.frame(ev_clean_pca_03$x[,1:2])
total_data_3 <- data.frame(
  PC1 = pca_scores_2$PC1,
  PC2 = pca_scores_2$PC2
)

#Regresion lineal

lm_pca_0 <- lm(PC2 ~ PC1, total_data)
summary(lm_pca_0)

#Explicación de la variable EV_share utilizando PC1 y PC2
total_data_4 <- data.frame(
  PC1 = pca_scores_2$PC1,
  PC2 = pca_scores_2$PC2,
  Objective_2 = ev_clean_03$EV.Share....
)



#Primera regresión lineal. EV.Share vs PC1

lm_pca_01 <- lm(Objective_2 ~ PC1, total_data_4)
summary(lm_pca_01)

##Comprobación de los supuestos necesarios para aplicar modelos de regresión lineal

#Comprobación de especificación de modelo correcta
resettest(lm_pca_01)

#Comprobación de linealidad de los erroes
shapiro.test(lm_pca_01$residuals)
# Gráfico de Cuantiles Teóricos (Q-Q Plot)
# Si los puntos siguen la línea recta, los residuos son normales.
qqnorm(lm_pca_01$residuals)
qqline(lm_pca_01$residuals, col = "red")

# Histograma de los residuos
# Para ver la forma de la distribución
hist(lm_pca_01$residuals, col = "lightblue")

#Comprobación de media 0 de los errores
mean(lm_pca_01$residuals)
t.test(lm_pca_01$residuals, alternative = "two.sided", mu=0)

#comprobación de homoscedasticidad de los residuos
bptest(lm_pca_01)
# Muestra el gráfico de Residuos vs. Ajustados
par(mfrow = c(1,2))
plot(lm_pca_01, which = 1)
plot(lm_pca_01, which = 3)

#comprobación de independencia de los errores
durbinWatsonTest(lm_pca_01)

#Dibujo intervalos de confianza y predicción.
# 2. Calcular ambos intervalos (confianza y predicción) al 95%
intervalo_confianza <- predict(lm_pca_01, interval = "confidence", level = 0.95)
colnames(intervalo_confianza)[1:3] <- c("fit","conf_lwr","conf_upr")

intervalo_prediccion <- predict(lm_pca_01, interval = "prediction", level = 0.95)
intervalo_prediccion <- intervalo_prediccion[,-1]
colnames(intervalo_prediccion)[1:2] <- c("pred_lwr","pred_upr")

# 3. Añadir estos intervalos a nuestro data frame original
total_data_con_intervalos <- cbind(total_data_4, intervalo_confianza, intervalo_prediccion)


# Vemos cómo ha quedado el data frame
head(total_data_con_intervalos)

#Creamos el gráfico con ggplot
ggplot(total_data_con_intervalos, aes(x = PC1, y = Objective_2)) +
  
  # Capa 1: Intervalo de Predicción (la banda más ancha)
  geom_ribbon(aes(ymin = pred_lwr, ymax = pred_upr), 
              fill = "green", 
              alpha = 0.2) +
  
  # Capa 2: Intervalo de Confianza (la banda más estrecha)
  geom_ribbon(aes(ymin = conf_lwr, ymax = conf_upr), 
              fill = "red", 
              alpha = 0.4) +
  
  # Capa 3: La recta de regresión
  geom_line(aes(y = fit), color = "darkred", size = 1) +
  
  # Capa 4: La nube de puntos
  geom_point(alpha = 0.6, color = "steelblue") +
  
  labs(
    title = "Regresión con Intervalos de Confianza y Predicción",
    subtitle = "Banda ancha = Predicción (95%), Banda estrecha = Confianza (95%)",
    x = "Componente Principal 1 (PC1)",
    y = "Variable Objetivo (Objective_2)"
  ) +
  theme_minimal()


###############################################################################
#Segunda regresión lineal. EV.Share vs PC1+PC2

lm_pca_02 <- lm(Objective_2 ~ PC1+PC2, total_data_4)
summary(lm_pca_02)

#Dibujo la nube de puntos y la recta de regresion de EV_Share frente a PC1

ggplot(total_data_4, aes(x = PC1, y = Objective_2)) +
  
  # Añade la nube de puntos
  geom_point(alpha = 0.6, color = "steelblue") +
  
  # Añade la recta de regresión lineal con su intervalo de confianza (la sombra gris)
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  
  labs(
    title = "Regresión de Objective_2 sobre PC1",
    x = "Componente Principal 1 (PC1)",
    y = "Variable Objetivo (Objective_2)"
  ) +
  theme_minimal()

#Dibujo la nube de puntos y la recta de regresion de EV_Share frente a PC2

ggplot(total_data_4, aes(x = PC2, y = Objective_2)) +
  
  # Añade la nube de puntos
  geom_point(alpha = 0.6, color = "darkgreen") +
  
  # Añade la recta de regresión lineal
  geom_smooth(method = "lm", color = "orange", se = TRUE) +
  
  labs(
    title = "Regresión de Objective_2 sobre PC2",
    x = "Componente Principal 2 (PC2)",
    y = "Variable Objetivo (Objective_2)"
  ) +
  theme_minimal()






###############################################################################
#Voy a quitar el outlier 'California' para ver como afecta

ev_clean_05 <- ev_clean_04[rownames(ev_clean_04) != "California",]
ev_clean_pca_04 <- prcomp(ev_clean_05, scale = TRUE)
summary(ev_clean_pca_04)

#Preparación de las variables para hacer una regresión lineal de los dos primeros componentes

pca_scores_3 <- as.data.frame(ev_clean_pca_04$x[,1:2])

total_data_5 <- data.frame(
  PC1 = pca_scores_3$PC1,
  PC2 = pca_scores_3$PC2,
  Objective_2 = ev_clean_03_reduced$EV.Share....
)
#Necesito utilizar ev_clean_03_reduced porque tiene información de EV.Share pero se ha quitado el estado de California

#Regresion lineal

lm_pca_3 <- lm(Objective_2 ~ PC1, total_data_5)
summary(lm_pca_3)

#Dibujo la nube de puntos y la recta de regresion de EV_Share frente a PC1

ggplot(total_data_5, aes(x = PC1, y = Objective_2)) +
  
  # Añade la nube de puntos
  geom_point(alpha = 0.6, color = "steelblue") +
  
  # Añade la recta de regresión lineal con su intervalo de confianza (la sombra gris)
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  
  labs(
    title = "Regresión de Objective_2 sobre PC1",
    x = "Componente Principal 1 (PC1)",
    y = "Variable Objetivo (Objective_2)"
  ) +
  theme_minimal()

#Dibujo intervalos de confianza y predicción.
# 2. Calcular ambos intervalos (confianza y predicción) al 95%
intervalo_confianza <- predict(lm_pca_3, interval = "confidence", level = 0.95)
colnames(intervalo_confianza)[1:3] <- c("fit","conf_lwr","conf_upr")

intervalo_prediccion <- predict(lm_pca_3, interval = "prediction", level = 0.95)
intervalo_prediccion <- intervalo_prediccion[,-1]
colnames(intervalo_prediccion)[1:2] <- c("pred_lwr","pred_upr")

# 3. Añadir estos intervalos a nuestro data frame original
total_data_con_intervalos <- cbind(total_data_5, intervalo_confianza, intervalo_prediccion)


# Vemos cómo ha quedado el data frame
head(total_data_con_intervalos)

#Creamos el gráfico con ggplot
ggplot(total_data_con_intervalos, aes(x = PC1, y = Objective_2)) +
  
  # Capa 1: Intervalo de Predicción (la banda más ancha)
  geom_ribbon(aes(ymin = pred_lwr, ymax = pred_upr), 
              fill = "red", 
              alpha = 0.2) +
  
  # Capa 2: Intervalo de Confianza (la banda más estrecha)
  geom_ribbon(aes(ymin = conf_lwr, ymax = conf_upr), 
              fill = "red", 
              alpha = 0.4) +
  
  # Capa 3: La recta de regresión
  geom_line(aes(y = fit), color = "darkred", size = 1) +
  
  # Capa 4: La nube de puntos
  geom_point(alpha = 0.6, color = "steelblue") +
  
  labs(
    title = "Regresión con Intervalos de Confianza y Predicción",
    subtitle = "Banda ancha = Predicción (95%), Banda estrecha = Confianza (95%)",
    x = "Componente Principal 1 (PC1)",
    y = "Variable Objetivo (Objective_2)"
  ) +
  theme_minimal()