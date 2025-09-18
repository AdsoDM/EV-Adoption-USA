#Clusterización

#Carga de librerías
library(factoextra)#visulalización para PCA. Contiene fviz 
library(cluster) #funciones de clusterización
library(ggplot2)#para construir el gráfico de afiliación política sobre los clusters
library(dplyr)#para construir el gráfico de afiliación política sobre los clusters
library(ggrepel) # Para etiquetas que no se solapen

#Preparación del dataset

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

#Creo una estructura de datos para trabajar

pca_scores <- as.data.frame(ev_clean_pca_01$x[,1:2])

cluster_data_01 <- scale(pca_scores)

#calculo de distancias

distance_01 <- get_dist(cluster_data_01)
view(as.matrix(distance_01))
fviz_dist(distance_01, gradient = list(low="#00AFBB",mid="white",high="#FC4E07"))

which(as.matrix(distance_01)==min(distance_01),arr.ind=T)

#modelado con  kmeans, 2 clusters

k2_01 <- kmeans (cluster_data_01, centers = 2)
sort(k2_01$cluster)

#Visualización
fviz_cluster(k2_01, data = cluster_data_01)

#VOy a ver los datos solamente del año 2023 porque si no es dificil observar nada

ev_clean_03 <- ev_clean_01 %>% filter(year==2023)
ev_clean_03 <- ev_clean_03 %>% column_to_rownames(var="state")
ev_clean_03 <- ev_clean_03 %>% select(-Index,-year)
ev_clean_03 <- ev_clean_03 %>% select(-fuel_economy)#elimino la variable fuel economy porque es una variable anual y su valor es constante para el mismo año


ev_clean_pca_02 <- prcomp(ev_clean_03, scale = TRUE)
summary(ev_clean_pca_02)

#Creo una estructura de datos para trabajar

pca_scores_02 <- as.data.frame(ev_clean_pca_02$x[,1:2])

cluster_data_02 <- scale(pca_scores_02)

#calculo de distancias

distance_02 <- get_dist(cluster_data_02)
view(as.matrix(distance_02))
fviz_dist(distance_02, gradient = list(low="#00AFBB",mid="white",high="#FC4E07"))

which(as.matrix(distance_01)==min(distance_01),arr.ind=T)
which(as.matrix(distance_01)==max(distance_01),arr.ind=T)

#modelado con  kmeans, 2 clusters

k2_02 <- kmeans (cluster_data_02, centers = 2)
sort(k2_02$cluster)

#Visualización
fviz_cluster(k2_02, data = cluster_data_02)#van cambiando los clusters con cada ejecución

###Voy a eliminar California ya que está 4 desviaciones típicas alejado de la media y distornsiona los resultados

cluster_data_03 <- cluster_data_02[rownames(cluster_data_02) != "California",]
k2_03 <- kmeans (cluster_data_03, centers = 2)
fviz_cluster(k2_03, data = cluster_data_03)#van cambiando los clusters con cada ejecución

#Voy a superponer la afiliación politica
ev_clean_03_reduced <- ev_clean_03[rownames(ev_clean_03) != "California",]
# Cargar las librerías necesarias
library(ggplot2)
library(dplyr)
library(ggrepel) # Para etiquetas que no se solapen

# --- PASO 1: Crear el Data Frame para el Gráfico ---
# Este data frame unirá toda la información que necesitamos.

# Asumimos que estos objetos ya existen de tus pasos anteriores:
# - cluster_data_03: La matriz con los scores de PC1 y PC2 (sin California)
# - k2_03: El resultado de tu función kmeans()
# - ev_clean_03_reduced: Tu data frame con la información original (sin California)

plot_data <- as.data.frame(cluster_data_03) %>%
  # Añadimos una columna con el clúster asignado por kmeans. Lo convertimos a factor.
  mutate(Cluster = factor(k2_03$cluster)) %>%
  
  # Añadimos una columna con la afiliación política, convertida a factor con etiquetas claras.
  mutate(Party = factor(ev_clean_03_reduced$Party,
                        levels = c(0, 1),
                        labels = c("Republican", "Democrat"))) %>%
  
  # Añadimos los nombres de los estados para poder etiquetarlos.
  mutate(State = rownames(cluster_data_03))


# --- PASO 2: Construir el Gráfico con ggplot2 ---

# Ahora creamos el gráfico capa por capa
ggplot(plot_data, aes(x = PC1, y = PC2, color = Party, shape = Cluster)) +
  
  # Capa 1: Dibuja los puntos. 'color' y 'shape' se asignan automáticamente.
  geom_point(size = 4, alpha = 0.8) +
  
  # Capa 2 (Opcional pero recomendado): Añade elipses para los clústeres
  stat_ellipse(aes(group = Cluster, color = NULL), linetype = "dashed", type = "norm") +
  
  # Capa 3 (Opcional): Añade etiquetas de texto para los estados que no se solapan.
  geom_text_repel(aes(label = State), 
                  color = "black", # Color del texto
                  size = 3, 
                  show.legend = FALSE) +
  
  # --- Personalización y Títulos ---
  
  # Personaliza los colores para que coincidan con los partidos
  scale_color_manual(values = c("Republican" = "red", "Democrat" = "blue")) +
  
  # Personaliza las formas (opcional, 16=círculo, 17=triángulo)
  scale_shape_manual(values = c("1" = 16, "2" = 17)) +
  
  labs(
    title = "Análisis de Clústeres y Afiliación Política",
    subtitle = "Color = Partido, Forma = Clúster Asignado",
    x = "Componente Principal 1 (PC1)",
    y = "Componente Principal 2 (PC2)",
    color = "Afiliación Política",
    shape = "Clúster (K-Means)"
  ) +
  theme_minimal() +
  # Añade una línea de referencia en el origen
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey")
