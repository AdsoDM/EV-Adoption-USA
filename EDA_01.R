#Importación de librerías
library(fastDummies) #Necesaria para el one hot encoding
  #Reducción dimensional
library(ggplot2)
library(ggfortify)
library(gridExtra)
library(factoextra)
library(corrplot)

#Cargo los datasets con los que voy a trabajar

training_01 <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/training_clean.csv")
test_01 <- read.csv(file="E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/test_clean.csv")

#Codificación de la variable Party. Asigno valor 1 al partido demócrata porque sus políticas sulen ir más alineadas con el ecologismo y la lucha contra el cambio climático

training_01 <- training_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))
test_01 <- test_01 %>% mutate(Party=factor(Party, levels=c('Republican','Democratic'), labels=c(0,1)))

#la nueva columna es de tipo factor y tengo que pasarla primero a caractery luego a número porque si no da error

training_01 <- training_01 %>% mutate(Party = as.numeric(as.character(Party)))
test_01 <- test_01 %>% mutate(Party = as.numeric(as.character(Party)))

# Identificar programáticamente cuáles columnas en el training set son numéricas
#    (Excluimos columnas de ID si no son predictoras)
columnas_numericas <- names(training_01)[sapply(training_01, is.numeric)]
columnas_no_numericas <- names(training_01[!sapply(training_01, is.numeric)])

#Codificamos los estados añadiendo 51 columnas (eliminamos una de las columnas para evitar la multicolinealidad perfecta)
#Utilizamos el método fast dummies

training_coded <- dummy_cols(
  .data = training_01,
  select_columns = "state",
  remove_selected_columns = TRUE,
  remove_first_dummy = TRUE
)

test_coded <- dummy_cols(
  .data = test_01,
  select_columns = "state",
  remove_selected_columns = TRUE,
  remove_first_dummy = TRUE
)


#scalado de características
#La función scale devuelve una matriz. Convierto a data frame
training_z01 <- as.data.frame(scale(training_coded))
test_z01 <- as.data.frame(scale(test_coded))
 
#Dibujo la matriz de correlación

training_z01 %>% 
  select(year:Party) %>%
  correlate() %>%
  plot()

#Creo un pdf con los boxplots de cada variable numérica

print_boxplots(training_01,"E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/boxplots.pdf")

eda_web_report(training_01)


#Análisis de reducción dimensional

#escalado de características sin la variable index y dejando fuera la variable stado, que será lo que voy a utilizar para analizar los resultados del PCA

#Voy a eliminar la columna index que no aporta nada al análisis

training_02 <- training_01 %>% select_if(is.numeric)
training_02 <- training_02 %>% select(-Index,-year)

#Voy a crear nombres de filas usando el año y el estado del dataset training_01

row_names <- paste(training_01$state,training_01$year, sep="_")
row.names(training_02) <- row_names

training_pca_01 <- prcomp(training_02, scale = TRUE)
summary(training_pca_01)

#análisis de resultados de pca

fviz_eig(training_pca_01, col.var="blue")
fviz_pca_var(training_pca_01,
             col.var = "cos2",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = TRUE)

ind <- get_pca_
fviz_pca_ind(training_pca_01,
             col.var = "cos2",
             gradient.cols = c("darkorchid4","gold","darkorange"),
             repel = FALSE)
