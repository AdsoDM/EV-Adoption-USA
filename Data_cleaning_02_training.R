#Práctica Final_v01_EV Adoption USA

#Carga de librerías
#Data Cleaning
library(caTools)#necesario para crear el knife para separar datos entre train y test
library(dplyr) # incluido en tidyverse. cargando tidyverse tendríamos acceso a todas las herramientas necesarias
library(fastDummies)#necesario para crear variables Dummy para convertir a números variables catégoricas
library(ggplot2)
library(dlookr)#necesario para las pruebas de normalidad
library(VIM)#necesario para la imputación de mazo
library(FNN)#necesario para imputación por KNN

#Cargo mis funciones
source("E:/Oscar/DataScience/Kryterion/Trabajo final/adhoc_functions.R")

#Carga de datos
ev_adoption_01 <- read.csv("E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/EV_data.csv")
#Elimino la la primera columna por esta duplicada y renombro la segunda a index



#Data Wrangling
#La tabla cumple con la 3º forma normal de Cood y no es necesario aplicar Data Wrangling
#la 3º forma normal de Cood exige que:
# (1) Cada variable forma una columna
# (2) Cada observación forma una fila
# (3) Cada unidad observacional forma una tabla

#Elimino la la primera columna por esta duplicada y renombro la segunda a index
ev_adoption_01 <- ev_adoption_01[,-1]
colnames(ev_adoption_01)[1]<- "Index"


#Data cleaning
ev_adoption_01_BU <- ev_adoption_01 #Hago una copia de backUp del dataframe
str(ev_adoption_01)

#Analizamos los valores faltantes

ina <- is.na(unlist(ev_adoption_01)) #crea un vector con los valores 'not available' del dataset
sum(ina)

#Comprobamos cuantas instancias estan completas (no tienen datos faltantes)
sum(complete.cases(ev_adoption_01))

#Tiene sentido dividir los datos desde un 50/50 hasta un 80/20 (training/test)
#Voy a hacer el split segregando previamente por estado

# 1. Asegurar que la división sea reproducible
set.seed(123) # ¡Importante para que siempre obtengas el mismo split!

# 2. Crear el conjunto de ENTRENAMIENTO (80%)
#    Agrupamos por estado y luego tomamos el 80% de las filas de CADA estado.
training_01 <- ev_adoption_01 %>%
  group_by(state) %>%
  slice_sample(prop = 0.70) %>%
  ungroup() # Desagrupamos como buena práctica después de la operación

# Guardo el objeto 'training_set' en un archivo llamado "conjunto_de_training.rds"
# El archivo se guardará en tu directorio de trabajo actual.
saveRDS(training_01, file = "conjunto_de_training.rds")

# 3. Crear el conjunto de PRUEBA (el 20% restante)
#    La forma más segura de obtener el resto es seleccionar las filas del dataset
#    original que NO están en el training_set. Usamos anti_join() para esto.
#    Necesitamos una columna de ID única para que funcione bien. Usaremos la 
#    columna 'Index' que vimos que tenías en tu dataset.
test_01 <- ev_adoption_01 %>%
  anti_join(training_01, by = "Index")

# Guardo el objeto 'test_set' en un archivo llamado "conjunto_de_test.rds"
# El archivo se guardará en tu directorio de trabajo actual.
saveRDS(test_01, file = "conjunto_de_test.rds")


# 4. VERIFICAR LOS RESULTADOS (Paso Opcional pero Muy Recomendado)

# Comprobar las dimensiones de los nuevos datasets
print(paste("Total de filas en Training Set:", nrow(training_set)))
print(paste("Total de filas en Test Set:", nrow(test_set)))

# Contar cuántas filas hay por estado en el conjunto de entrenamiento
# Verás que para cada estado hay aproximadamente el 80% de sus datos originales.
# Si un estado tenía 8 filas, aquí tendrá 6 o 7.
print("Conteo de filas por estado en el training Set:")
conteo_training <- training_set %>% count(state)
as.data.frame(conteo_training) #convierto a data frame para ver todos los valores
print(conteo_training)

# Contar cuántas filas hay por estado en el conjunto de prueba
# Verás que cada estado tiene las filas restantes.
# Si un estado tenía 8 filas, aquí tendrá 1 o 2.
print("Conteo de filas por estado en el Test Set:")
conteo_test <- test_set %>% count(state)
as.data.frame(conteo_test) #convierto a data frame para ver todos los valores
print(conteo_test)

#Tratamiento de datos faltantes

print_atributes(training_01)

#Análisis de imputaciones

# 1. Crear un vector vacío para guardar los resultados
conteo_nas_por_columna <- c()


#Bucle para obtner los valores faltantes por variable
for (nombre_columna in names(training_01)) {
  # 3. Para cada columna, contar los NAs y guardar el resultado
  #    is.na() crea un vector de TRUE/FALSE. sum() cuenta los TRUE como 1 y los FALSE como 0.
  conteo_na <- sum(is.na(training_01[[nombre_columna]]))
  
  # 4. Añadir el resultado al vector de resultados
  conteo_nas_por_columna[nombre_columna] <- conteo_na
}

print(conteo_nas_por_columna)

#Variable Fuel economy

#dibujamos la evolución de la variable 'fuel_economy'
#La variable 'fuel_economy' es el consumo medio en todo el pais (los 51 estados cada año). Faltan dos años completos de datos, el 2016 y el 2017.
ggplot(data = training_01, aes(x=Index, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Index", y="Consumo medio de los vehículos del estado")
ggplot(data = training_01, aes(x=year, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Year", y="Consumo medio de los vehículos del estado")

#Fuel_economy
#dibujamos la evolución de la población según fecha
#Contamos las repeticiones de los diferentes valores
training_01 %>%  count(fuel_economy, sort = TRUE)

#dibujamos la evolución de la variable 'fuel_economy'
ggplot(data = training_01, aes(x=year, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Year", y="Consumo medio de combustible de los vehículos de EEUU")
ggplot(data = training_01, aes(x=Index, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Index", y="Consumo medio de combustible de los vehículos de EEUU")
ggplot(data = training_01, aes(x=fuel_economy))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Consumo medio de combustible de los vehículos de EEUU", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()

#Compruebo normalidad
training_01 %>% plot_normality(fuel_economy) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_01$fuel_economy)
print(resultado_shapiro)

#Voy a imputar por regresion lineal
# Aplicar la función de imputación por regresión a cada estado
# Asumo que la columna de precios se llama 'gasoline_price_per_gallon'
#Voy a hacer la regresion con los valores del 2018 al 2020


training_01 <- training_01 %>%
  # Usamos mutate para reemplazar la columna de precios con la versión imputada
  mutate(
    fuel_economy = imputar_regresion_filtrada(
      valores = fuel_economy,
      anios = year,
      anios_modelo = c(2018,2019,2020)
    )
  )

# Verificar el resultado para un estado y los años de interés
# Deberías ver que los valores de 2016 y 2017 ahora están rellenos
# con valores que siguen la tendencia de los años posteriores.
training_01 %>%
  filter(state == "California", year %in% 2016:2019) %>%
  select(state, year, fuel_economy)

#dibujamos la evolución de la variable 'fuel_economy'
ggplot(data = training_01, aes(x=year, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Year", y="Consumo medio de combustible de los vehículos de EEUU")
ggplot(data = training_01, aes(x=Index, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Index", y="Consumo medio de combustible de los vehículos de EEUU")
ggplot(data = training_01, aes(x=fuel_economy))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Consumo medio de combustible de los vehículos de EEUU", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()



#Incentives
ggplot(data = training_01, aes(x=year, y = Incentives))+ geom_line(aes(color = Incentives))+labs(x="Year", y="Presencia de incentivos por parte del Estado")
ggplot(data = training_01, aes(x=Incentives))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Distribución de los incentivos a vehículos eléctricos", x="Número de incentivos", y="Frecuencia") +
  theme_minimal()

#Voy a probar a imputar por media
training_02 <- training_01
training_02$Incentives = ifelse(is.na(training_01$Incentives), ave(training_01$Incentives, FUN = function(x) mean(x, na.rm = TRUE)), training_01$Incentives)

#Number.of.Metro.Organizing.Committees

ggplot(data = training_01, aes(x=year, y = Number.of.Metro.Organizing.Committees))+ geom_line(aes(color = Number.of.Metro.Organizing.Committees))+labs(x="Year", y="Número de comités de organización metropolitano")
ggplot(data = training_01, aes(x=Index, y = Number.of.Metro.Organizing.Committees))+ geom_line(aes(color = Number.of.Metro.Organizing.Committees))+labs(x="Index", y="Número de comités de organización metropolitano")
ggplot(data = training_01, aes(x=Number.of.Metro.Organizing.Committees))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Número de comités de organización metropolitanos", x="Número de incentivos", y="Frecuencia") +
  theme_minimal()

#Faltan los datos de un año . Voy a imputar como la media del año anterior y el posterior por cada estado.

# Imputación en un solo paso
training_03 <- training_02 %>%
  
  # 1. Agrupar por estado para que los cálculos se hagan por separado para cada uno
  group_by(state) %>%
  # 2. Usar mutate para crear la media y luego usarla para imputar
  mutate(
    # PRIMERO: Calculamos la media de 2019-2020 y la guardamos en una columna temporal.
    # Esta media será la misma para todas las filas de un mismo estado.
    media_para_imputar = mean(Number.of.Metro.Organizing.Committees[year %in% c(2018, 2020,2021)], na.rm = TRUE),
    # SEGUNDO: Modificamos la columna original usando la columna temporal que acabamos de crear.
    Number.of.Metro.Organizing.Committees = ifelse(
      # Condición: Si el año es 2018 y el valor es NA...
      year == 2019 & is.na(Number.of.Metro.Organizing.Committees),
      # ...entonces, usa la media que calculamos en el paso anterior.
      round(media_para_imputar),
      # ...de lo contrario, deja el valor original.
      Number.of.Metro.Organizing.Committees
    )
  ) %>%
  
  # 3. Desagrupar, una buena práctica después de group_by y mutate
  ungroup() %>%
  
  # 4. Eliminar la columna temporal que ya no necesitamos
  select(-media_para_imputar)

# Verificar el resultado (debería ser idéntico al del método anterior)
training_03 %>%
  filter(state == "Alabama", year %in% c(2018, 2019, 2020)) %>%
  select(state, year, Number.of.Metro.Organizing.Committees)
    
##Affectweather
ggplot(data = training_01, aes(x=year, y = affectweather))+ geom_line(aes(color = affectweather))+labs(x="Year", y="Medida de sensibilidad o precupacion por el cambio climático")
ggplot(data = training_01, aes(x=Index, y = affectweather))+ geom_line(aes(color = affectweather))+labs(x="Index", y="Medida de sensibilidad o precupacion por el cambio climático")
ggplot(data = training_01, aes(x=affectweather))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de sensibilidad o precupacion por el cambio climático", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()

training_03 %>% plot_normality(affectweather) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_03$affectweather)
print(resultado_shapiro)

#Voy a probar a imputar por media
training_04 <- training_03
#He probado a hacer una imputación por media y no me gusta el resultado. La variable ha perdido sus propiedades normales
training_04$affectweather = ifelse(is.na(training_04$affectweather), ave(training_04$affectweather, FUN = function(x) mean(x, na.rm = TRUE)), training_04$affectweather)

#Voy a probar a hacer una imputacion de mazo
set.seed(123) # De nuevo, para reproducibilidad

# Realizar la imputación estratificada por estado ('state')
training_04 <- hotdeck(
  training_04,
  variable = "affectweather",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

# Puedes incluso estratificar por más de una variable
# set.seed(123)
# training_imputado_estratificado_2 <- hotdeck(
#   training_01,
#   variable = "affectweather",
#   domain_var = c("state", "Party") # Usa donantes del mismo estado y partido
# )

#Compruebo normalidad
training_04 %>% plot_normality(affectweather) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_04$affectweather)
print(resultado_shapiro)

#Se ha empeorado ligeramente la normalidad pero sigue siendo una hipótesis aceptable
#Ya hemos pasado de 1505 datos faltantes a 1344
ina <- is.na(unlist(training_04)) #crea un vector con los valores 'not available' del dataset
sum(ina)

#Variable DEVHarm

ggplot(data = training_04, aes(x=year, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Year", y="Medida de sensibilidad o precupacion por el impacto del cambio climatico en el crecimiento económico")
ggplot(data = training_04, aes(x=Index, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Index", y="Medida de sensibilidad o precupacion por el impacto del cambio climatico en el crecimiento económico")
ggplot(data = training_04, aes(x=devharm))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de sensibilidad o precupacion por el cambio climático", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()

#Compruebo normalidad
training_04 %>% plot_normality(devharm) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_04$devharm)
print(resultado_shapiro)

#Voy a probar a hacer una imputacion de mazo
set.seed(123) # De nuevo, para reproducibilidad

#Los datos tienen una tendencia temporal.
#La preocupación va aumentando con los años.Por lo tanto voy a implementar una imputacion de mazo pero cogiendo valores solo de los años 2018 y 2019
# Aplicar la función de imputación a cada estado
training_05 <- training_04 %>%
  group_by(state) %>%  # Agrupamos por estado para que la función se aplique a cada uno por separado
  mutate(
    devharm = imputar_con_tendencia(devharm, year, c(2018,2019,2020,2021,2022,2023),c(2016,2017)) # Aplicamos nuestra función a las columnas devharm y year
  ) %>%
  ungroup() # Desagrupamos como buena práctica

# Verificar el resultado para un estado y los años de interés
training_05 %>%
  filter(state == "Alabama", year %in% 2016:2019) %>%
  select(state, year, devharm)

#Compruebo normalidad y plots
ggplot(data = training_05, aes(x=year, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Year", y="Medida de sensibilidad o precupacion por el impacto del cambio climatico en el crecimiento económico")
ggplot(data = training_05, aes(x=Index, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Index", y="Medida de sensibilidad o precupacion por el impacto del cambio climatico en el crecimiento económico")
ggplot(data = training_05, aes(x=devharm))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de sensibilidad o precupacion por el cambio climático", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()
training_05 %>% plot_normality(devharm) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_05$devharm)
print(resultado_shapiro)
#Se pierde normalidad pero ya se ve que hay una tendencia temporal y por tanto los datos no son normales de partida

#Variable Discuss


ggplot(data = training_05, aes(x=state, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Year", y="Medida de la frecuencia con la que la gente discute sobre asuntos medioambientales")
ggplot(data = training_05, aes(x=Index, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Index", y="Medida de la frecuencia con la que la gente discute sobre asuntos medioambientales")
ggplot(data = training_05, aes(x=devharm))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la frecuencia con la que la gente discute sobre asuntos medioambientales", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()
training_05 %>% plot_normality(discuss) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_05$discuss)
print(resultado_shapiro)

# Realizar la imputación estratificada por estado ('state')
training_06 <- hotdeck(
  training_05,
  variable = "discuss",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

#Compruebo normalidad y plots
ggplot(data = training_06, aes(x=state, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Year", y="Medida de la frecuencia con la que la gente discute sobre asuntos medioambientales")
ggplot(data = training_06, aes(x=Index, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Index", y="Medida de la frecuencia con la que la gente discute sobre asuntos medioambientales")
ggplot(data = training_06, aes(x=discuss))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la frecuencia con la que la gente discute sobre asuntos medioambientales", x="Sensibilidad", y="Frecuencia") +
  theme_minimal()
training_06 %>% plot_normality(discuss) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_06$discuss)
print(resultado_shapiro)

#Variable exp
##
ggplot(data = training_06, aes(x=state, y = exp))+ geom_line(aes(color = exp))+labs(x="state", y="Medida de la experiencia y conocimiento sobre asuntos medioambientales")
ggplot(data = training_06, aes(x=year, y = exp))+ geom_line(aes(color = exp))+labs(x="Year", y="Medida de la experiencia y conocimiento sobre asuntos medioambientales")
ggplot(data = training_06, aes(x=Index, y = exp))+ geom_line(aes(color = exp))+labs(x="Index", y="Medida de la experiencia y conocimiento sobre asuntos medioambientales")
ggplot(data = training_06, aes(x=exp))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la experiencia y conocimiento sobre asuntos medioambientales", x="Conocimiento", y="Frecuencia") +
  theme_minimal()
training_06 %>% plot_normality(exp) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_06$exp)
print(resultado_shapiro)

## Realizar la imputación estratificada por estado ('state')
training_07 <- hotdeck(
  training_06,
  variable = "exp",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

#Compruebo normalidad y grafico
ggplot(data = training_07, aes(x=state, y = exp))+ geom_line(aes(color = exp))+labs(x="state", y="Medida de la experiencia y conocimiento sobre asuntos medioambientales")
ggplot(data = training_07, aes(x=year, y = exp))+ geom_line(aes(color = exp))+labs(x="Year", y="Medida de la experiencia y conocimiento sobre asuntos medioambientales")
ggplot(data = training_07, aes(x=Index, y = exp))+ geom_line(aes(color = exp))+labs(x="Index", y="Medida de la experiencia y conocimiento sobre asuntos medioambientales")
ggplot(data = training_07, aes(x=exp))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la experiencia y conocimiento sobre asuntos medioambientales", x="Conocimiento", y="Frecuencia") +
  theme_minimal()
training_07 %>% plot_normality(exp) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_07$exp)
print(resultado_shapiro)

#Variable localofficials
#
ggplot(data = training_07, aes(x=state, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="state", y="Medida de la confianza en las oficinas locales de medioambiente")
ggplot(data = training_07, aes(x=year, y = localofficials))+ geom_line(aes(color = exp))+labs(x="Year", y="Medida de la confianza en las oficinas locales de medioambiente")
ggplot(data = training_07, aes(x=Index, y = localofficials))+ geom_line(aes(color = exp))+labs(x="Index", y="Medida de la confianza en las oficinas locales de medioambiente")
ggplot(data = training_07, aes(x=localofficials))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la confianza en las oficinas locales de medioambiente", x="Conocimiento", y="Frecuencia") +
  theme_minimal()
training_07 %>% plot_normality(localofficials) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_07$localofficials)
print(resultado_shapiro)


## Realizar la imputación estratificada por estado ('state')
training_08 <- hotdeck(
  training_07,
  variable = "localofficials",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

#Compruebo normalidad y grafico
ggplot(data = training_08, aes(x=state, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="state", y="Medida de la confianza en las oficinas locales de medioambiente")
ggplot(data = training_08, aes(x=year, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="Year", y="Medida de la confianza en las oficinas locales de medioambiente")
ggplot(data = training_08, aes(x=Index, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="Index", y="Medida de la confianza en las oficinas locales de medioambiente")
ggplot(data = training_08, aes(x=localofficials))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la confianza en las oficinas locales de medioambiente", x="Conocimiento", y="Frecuencia") +
  theme_minimal()
training_08 %>% plot_normality(localofficials) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_08$localofficials)
print(resultado_shapiro)

#Variable personal
ggplot(data = training_08, aes(x=state, y = personal))+ geom_line(aes(color = personal))+labs(x="state", y="Medida de la responsabilidad personal hacia el medioambiente")
ggplot(data = training_08, aes(x=year, y = personal))+ geom_line(aes(color = personal))+labs(x="Year", y="Medida de la responsabilidad personal hacia el medioambiente")
ggplot(data = training_08, aes(x=Index, y = personal))+ geom_line(aes(color = personal))+labs(x="Index", y="Medida de la responsabilidad personal hacia el medioambiente")
ggplot(data = training_08, aes(x=personal))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la responsabilidad personal hacia el medioambiente", x="Responsabilidad", y="Frecuencia") +
  theme_minimal()
training_08 %>% plot_normality(personal) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_08$personal)
print(resultado_shapiro)

## Realizar la imputación estratificada por estado ('state')
training_09 <- hotdeck(
  training_08,
  variable = "personal",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

#Compruebo normalidad y grafico
ggplot(data = training_09, aes(x=state, y = personal))+ geom_line(aes(color = personal))+labs(x="state", y="Medida de la responsabilidad personal hacia el medioambiente")
ggplot(data = training_09, aes(x=year, y = personal))+ geom_line(aes(color = personal))+labs(x="Year", y="Medida de la responsabilidad personal hacia el medioambiente")
ggplot(data = training_09, aes(x=Index, y = personal))+ geom_line(aes(color = personal))+labs(x="Index", y="Medida de la responsabilidad personal hacia el medioambiente")
ggplot(data = training_09, aes(x=personal))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la responsabilidad personal hacia el medioambiente", x="Responsabilidad", y="Frecuencia") +
  theme_minimal()
training_09 %>% plot_normality(personal) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_09$personal)
print(resultado_shapiro)

#Variable reducetax
#
ggplot(data = training_09, aes(x=state, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="state", y="Medida del apoyo para reducir impuestos con políticas medioambientales")
ggplot(data = training_09, aes(x=year, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Year", y="Medida del apoyo para reducir impuestos con políticas medioambientales")
ggplot(data = training_09, aes(x=Index, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Index", y="Medida del apoyo para reducir impuestos con políticas medioambientales")
ggplot(data = training_09, aes(x=reducetax))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida del apoyo para reducir impuestos con políticas medioambientales", x="Reduccion tasas", y="Frecuencia") +
  theme_minimal()
training_09 %>% plot_normality(reducetax) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_09$reducetax)
print(resultado_shapiro)

## Realizar la imputación estratificada por estado ('state')
training_10 <- hotdeck(
  training_09,
  variable = "reducetax",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

#Compruebo normalidad y grafico
ggplot(data = training_10, aes(x=state, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="state", y="Medida del apoyo para reducir impuestos con políticas medioambientales")
ggplot(data = training_10, aes(x=year, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Year", y="Medida del apoyo para reducir impuestos con políticas medioambientales")
ggplot(data = training_10, aes(x=Index, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Index", y="Medida del apoyo para reducir impuestos con políticas medioambientales")
ggplot(data = training_10, aes(x=reducetax))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida del apoyo para reducir impuestos con políticas medioambientales", x="Reduccion tasas", y="Frecuencia") +
  theme_minimal()
training_10 %>% plot_normality(reducetax) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_10$reducetax)
print(resultado_shapiro)

#Variable regulate
#
ggplot(data = training_10, aes(x=state, y = regulate))+ geom_line(aes(color = regulate))+labs(x="state", y="Medida del apoyo del gobierno a regulaciones medioambientales")
ggplot(data = training_10, aes(x=year, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Year", y="Medida del apoyo del gobierno a regulaciones medioambientales")
ggplot(data = training_10, aes(x=Index, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Index", y="Medida del apoyo del gobierno a regulaciones medioambientales")
ggplot(data = training_10, aes(x=regulate))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida del apoyo del gobierno a regulaciones medioambientales", x="Reduccion tasas", y="Frecuencia") +
  theme_minimal()
training_10 %>% plot_normality(regulate) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_10$regulate)
print(resultado_shapiro)

#Utilizo imputación de mazo segregando por estado y determinados años
#La preocupación va aumentando con los años.Por lo tanto voy a implementar una imputacion de mazo pero cogiendo valores solo de los años 2018 y 2019
# Aplicar la función de imputación a cada estado
training_11 <- training_10 %>%
  group_by(state) %>%  # Agrupamos por estado para que la función se aplique a cada uno por separado
  mutate(
    regulate = imputar_con_tendencia(regulate, year, c(2018,2019,2020,2021,2022,2023), c(2016,2017)) # Aplicamos nuestra función a las columnas devharm y year
  ) %>%
  ungroup() # Desagrupamos como buena práctica

# Verificar el resultado para un estado y los años de interés
training_11 %>%
  filter(state == "Alabama", year %in% 2016:2019) %>%
  select(state, year, regulate)

#Compruebo normalidad y grafico
ggplot(data = training_11, aes(x=state, y = regulate))+ geom_line(aes(color = regulate))+labs(x="state", y="Medida del apoyo del gobierno a regulaciones medioambientales")
ggplot(data = training_11, aes(x=year, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Year", y="Medida del apoyo del gobierno a regulaciones medioambientales")
ggplot(data = training_11, aes(x=Index, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Index", y="Medida del apoyo del gobierno a regulaciones medioambientales")
ggplot(data = training_11, aes(x=regulate))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida del apoyo del gobierno a regulaciones medioambientales", x="Apoyo de gobierno", y="Frecuencia") +
  theme_minimal()
training_11 %>% plot_normality(regulate) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_11$regulate)
print(resultado_shapiro)

#Varariable worried
#
ggplot(data = training_11, aes(x=state, y = worried))+ geom_line(aes(color = worried))+labs(x="state", y="Medida de la preocupación por problemas medioambientales")
ggplot(data = training_11, aes(x=year, y = worried))+ geom_line(aes(color = worried))+labs(x="Year", y="Medida de la preocupación por problemas medioambientales")
ggplot(data = training_11, aes(x=Index, y = worried))+ geom_line(aes(color = worried))+labs(x="Index", y="Medida de la preocupación por problemas medioambientales")
ggplot(data = training_11, aes(x=worried))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la preocupación por problemas medioambientales", x="Preocupacion", y="Frecuencia") +
  theme_minimal()
training_11 %>% plot_normality(worried) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_11$worried)
print(resultado_shapiro)

## Realizar la imputación estratificada por estado ('state')
training_12 <- hotdeck(
  training_11,
  variable = "worried",
  domain_var = "state" # La variable para crear los "estratos" o grupos
)

#Compruebo normalidad y grafico
ggplot(data = training_12, aes(x=state, y = worried))+ geom_line(aes(color = worried))+labs(x="state", y="Medida de la preocupación por problemas medioambientales")
ggplot(data = training_12, aes(x=year, y = worried))+ geom_line(aes(color = worried))+labs(x="Year", y="Medida de la preocupación por problemas medioambientales")
ggplot(data = training_12, aes(x=Index, y = worried))+ geom_line(aes(color = worried))+labs(x="Index", y="Medida de la preocupación por problemas medioambientales")
ggplot(data = training_12, aes(x=worried))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Medida de la preocupación por problemas medioambientales", x="Preocupacion", y="Frecuencia") +
  theme_minimal()
training_12 %>% plot_normality(worried) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_12$worried)
print(resultado_shapiro)

#Variable gasoline_price_per_gallon
#
ggplot(data = training_12, aes(x=state, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="state", y="Precio medio de la gasolina en el estado")
ggplot(data = training_12, aes(x=year, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Year", y="Precio medio de la gasolina en el estado")
ggplot(data = training_12, aes(x=Index, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Index", y="Precio medio de la gasolina en el estado")
ggplot(data = training_12, aes(x=gasoline_price_per_gallon))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Precio medio de la gasolina en el estado", x="Precio gasolina", y="Frecuencia") +
  theme_minimal()
training_12 %>% plot_normality(gasoline_price_per_gallon) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_12$gasoline_price_per_gallon)
print(resultado_shapiro)

#Voy a imputar por regresion lineal
# Aplicar la función de imputación por regresión a cada estado
# Asumo que la columna de precios se llama 'gasoline_price_per_gallon'
training_13 <- training_12 %>%
  
  # Agrupamos por estado para que la imputación se haga de forma independiente para cada uno
  group_by(state) %>%
  
  # Usamos mutate para reemplazar la columna de precios con la versión imputada
  mutate(
    gasoline_price_per_gallon = imputar_regresion(
      valores = gasoline_price_per_gallon,
      anios = year
    )
  ) %>%
  
  # Desagrupamos como buena práctica
  ungroup()

# Verificar el resultado para un estado y los años de interés
# Deberías ver que los valores de 2016 y 2017 ahora están rellenos
# con valores que siguen la tendencia de los años posteriores.
training_13 %>%
  filter(state == "California", year %in% 2016:2019) %>%
  select(state, year, gasoline_price_per_gallon)

#Comrpuebo normalidad y grafico
ggplot(data = training_13, aes(x=state, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="state", y="Precio medio de la gasolina en el estado")
ggplot(data = training_13, aes(x=year, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Year", y="Precio medio de la gasolina en el estado")
ggplot(data = training_13, aes(x=Index, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Index", y="Precio medio de la gasolina en el estado")
ggplot(data = training_13, aes(x=gasoline_price_per_gallon))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Precio medio de la gasolina en el estado", x="Precio gasolina", y="Frecuencia") +
  theme_minimal()
training_13 %>% plot_normality(gasoline_price_per_gallon) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_13$gasoline_price_per_gallon)
print(resultado_shapiro)

#Variable Trucks
#
ggplot(data = training_13, aes(x=state, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="state", y="Número de camiones")
ggplot(data = training_13, aes(x=year, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Year", y="Número de camiones")
ggplot(data = training_13, aes(x=Index, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Index", y="Número de camiones")
ggplot(data = training_13, aes(x=Trucks))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Número de camiones", x="Camiones", y="Frecuencia") +
  theme_minimal()
training_13 %>% plot_normality(Trucks) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_13$Trucks)
print(resultado_shapiro)

#Voy a imputar por regresion lineal
# Aplicar la función de imputación por regresión a cada estado
# Asumo que la columna de precios se llama 'gasoline_price_per_gallon'
training_14 <- training_13 %>%
  
  # Agrupamos por estado para que la imputación se haga de forma independiente para cada uno
  group_by(state) %>%
  
  # Usamos mutate para reemplazar la columna de precios con la versión imputada
  mutate(
    Trucks = imputar_hibrido_regresion(
      valores = Trucks,
      anios = year
    )
  ) %>%
  
  # Desagrupamos como buena práctica
  ungroup()

# Verificar el resultado para un estado y los años de interés
# Deberías ver que los valores de 2016 y 2017 ahora están rellenos
# con valores que siguen la tendencia de los años posteriores.
training_14 %>%
  filter(state == "New York", year %in% 2016:2019) %>%
  select(state, year, Trucks)

#Comrpuebo normalidad y grafico
ggplot(data = training_14, aes(x=state, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="state", y="Número de camiones")
ggplot(data = training_14, aes(x=year, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Year", y="Número de camiones")
ggplot(data = training_14, aes(x=Index, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Index", y="Número de camiones")
ggplot(data = training_14, aes(x=Trucks))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Número de camiones", x="Camiones", y="Frecuencia") +
  theme_minimal()
training_14 %>% plot_normality(Trucks) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_14$Trucks)
print(resultado_shapiro)

#El Distrito de Columbia no tiene datos, por lo tanto no es posible imputarlos mediante regresión.
#Hay que recurrir a la estrategia KNN (k-nearest neighbourgs)

#Paso 1: definir datos clave y estandarización

# 1. Preparar los datos para el análisis de similitud
#    Usaremos los datos del año más reciente y completo (ej. 2023) para definir la similitud estructural.
data_para_similitud <- training_14 %>%
  filter(year == 2023) %>%
  # Seleccionamos las columnas que definen la similitud
  select(state, Per_Cap_Income, Population_20_64, gasoline_price_per_gallon, Bachelor_Attainment, EV.Share....)


# 2. Estandarizar las variables numéricas (¡Paso CRÍTICO!)
#    Esto asegura que ninguna variable domine el cálculo de distancia solo por tener una escala más grande.
#    Guardamos los nombres de los estados y luego estandarizamos el resto.
states_list <- data_para_similitud$state
data_estandarizada <- data.frame(scale(select(data_para_similitud, -state)))
row.names(data_estandarizada) <- states_list

#Paso 2: Encontrar los K vecinos más próximos a District of Columbia

# 3. Separar los datos de D.C. del resto
dc_data_std <- data_estandarizada["District Of Columbia", ]
otros_estados_std <- data_estandarizada[row.names(data_estandarizada) != "District Of Columbia", ]

# 4. Encontrar los k vecinos más cercanos (vamos a usar k=5)
#    get.knnx busca los 'k' vecinos en 'otros_estados_std' para cada punto en 'dc_data_std'
k <- 5
vecinos <- get.knnx(data = otros_estados_std, 
                    query = dc_data_std, 
                    k = k)

# 5. Extraer los nombres de los estados vecinos
nombres_vecinos <- row.names(otros_estados_std)[vecinos$nn.index]
print(paste("Los", k, "estados más similares a D.C. son:", paste(nombres_vecinos, collapse=", ")))

#Paso 3: Imputar con la media de los 5 vecinos más cercanos

# 6. Calcular la media de los vecinos para cada año
imputacion_por_anio <- training_14 %>%
  filter(state %in% nombres_vecinos) %>% # Filtramos solo los datos de los estados vecinos
  group_by(year) %>%                     # Agrupamos por año
  summarise(
    trucks_imputado = mean(Trucks, na.rm = TRUE) # Calculamos la media de 'Trucks' para cada año
  )

print(imputacion_por_anio)

#Paso 3: Realizar la imputación con el vector calculado en el paso anterior
# 7. Unir estos valores de imputación a tu dataset original y rellenar los huecos
training_15 <- training_14 %>%
  # Unimos los valores a imputar basándonos en el año
  left_join(imputacion_por_anio, by = "year") %>%
  # Usamos mutate con ifelse para rellenar solo los NAs de D.C.
  mutate(
    Trucks = ifelse(
      state == "District Of Columbia" & is.na(Trucks), # Condición
      trucks_imputado,                                  # Valor si es TRUE
      Trucks                                            # Valor si es FALSE
    )
  ) %>%
  select(-trucks_imputado) # Limpiamos la columna de ayuda

# 8. Verificar el resultado para D.C.
training_15 %>%
  filter(state == "District Of Columbia") %>%
  select(state, year, Trucks)

#Comrpuebo normalidad y grafico
ggplot(data = training_15, aes(x=state, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="state", y="Número de camiones")
ggplot(data = training_15, aes(x=year, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Year", y="Número de camiones")
ggplot(data = training_15, aes(x=Index, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Index", y="Número de camiones")
ggplot(data = training_15, aes(x=Trucks))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Número de camiones", x="Camiones", y="Frecuencia") +
  theme_minimal()
training_15 %>% plot_normality(Trucks) #Es una función quasi-normal
resultado_shapiro <- shapiro.test(training_15$Trucks)
print(resultado_shapiro)

#variable Party
#
#Tengo ya el vector de vecinos más parecido a DC. Voy a aprovechar la información para imputar la columna Party por moda de los vecinos
#Como es una variable categórica dibujo solo el diagrama de barras
ggplot(data = training_15, aes(x=Party))+ 
  geom_bar(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Partido político en el poder", x="Partido", y="Frecuencia") +
  theme_minimal()

# Calcular la moda de los vecinos para cada año
training_16 <- training_15 %>%
  mutate(Party = na_if(Party,""))

imputacion_por_moda <- training_16 %>%
  filter(state %in% nombres_vecinos) %>% # Filtramos solo los datos de los estados vecinos
  group_by(year) %>%                     # Agrupamos por año
  summarise(
    party_imputado = getmode(Party) # Calculamos la moda de 'Party' para cada año
  )

print(imputacion_por_moda)

# Unir estos valores de imputación a tu dataset original y rellenar los huecos
training_16 <- training_16 %>%
  # Unimos los valores a imputar basándonos en el año
  left_join(imputacion_por_moda, by = "year") %>%
  # Usamos mutate con ifelse para rellenar solo los NAs de D.C.
  mutate(
    Party = ifelse(
      state == "District Of Columbia" & is.na(Party), # Condición
      party_imputado,                                  # Valor si es TRUE
      Party                                            # Valor si es FALSE
    )
  ) %>%
  select(-party_imputado) # Limpiamos la columna de ayuda

# Verificar el resultado para D.C.
training_16 %>%
  filter(state == "District Of Columbia") %>%
  select(state, year, Party)

#Ya he hecho las 6 imputaciones para el distrito de columbia que faltaban para los años 2018 a 2023.
#Ahora hay que imputar para todos los estados para los años 2016 y 2017. Voy a imputar por moda.

#Voy a probar a imputar por moda
training_16 <- training_16 %>%
  group_by(state) %>%
  mutate(
    Party= ifelse(
      is.na(Party),
      getmode(Party,na.rm=TRUE),
      Party
    )
  ) %>%
  ungroup()

#Como es una variable categórica dibujo solo el diagrama de barras
ggplot(data = training_16, aes(x=Party))+ 
  geom_bar(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Partido político en el poder", x="Partido", y="Frecuencia") +
  theme_minimal() 

#Limpieza final de la tabla
#elimino las columnas Trucks y Trucks_Share por redundancia o inconsistencia
borrar <- c("Total","Trucks_Share")
training_17 <- training_16[,!(names(training_16) %in% borrar)]

#elimino todas las columnas creadas por la imputacion de mazo

training_17 <- training_17 %>% select(Index:Party)

write.csv(training_17, file = "E:/Oscar/DataScience/Kryterion/Trabajo final/EV Adoption USA/data/training_clean.csv",row.names=FALSE)


#Las variables  gasoline_price_per_gallon y Trucks utilizan una regresión para imputar. POr como se han distribudio los datos en 
