#En este código incluiré algunas funciones para invocarlas posteriormete

# Calculo de la moda de un conjunto de datos

getmode <- function(v, na.rm = FALSE) {
  
  # Paso 1: Comprobar si el usuario quiere eliminar los valores NA.
  # Si na.rm es TRUE, se ejecuta el código dentro del if.
  if (na.rm) {
    # Filtra el vector, quedándose solo con los elementos que NO son NA.
    v <- v[!is.na(v)]
  }
  
  # Paso 2 (Opcional pero recomendado): Eliminar también cadenas de texto vacías ("").
  v <- v[nchar(as.character(v)) > 0]
  
  # Paso 3: Comprobación de seguridad.
  # Si el vector está vacío después de la limpieza, no hay moda que calcular.
  if (length(v) == 0) {
    # Devolvemos NA para indicar que no se pudo calcular la moda.
    # NA_character_ es un tipo específico de NA para vectores de texto.
    return(NA_character_) 
  }
  
  # Paso 4: Calcular la moda (la misma lógica que antes).
  # Esto se ejecuta sobre el vector 'v' ya limpio (si na.rm = TRUE).
  uniqv <- unique(v)
  return(uniqv[which.max(tabulate(match(v, uniqv)))])
}

#Función para crear un pdf con la representación gráfica de las distintas varibles de un dataset. 

print_atributes <- function(input_data){
  
  # Cargar las librerías es una buena práctica
  library(ggplot2)
  library(dplyr)
  
  # Abre un dispositivo PDF para guardar los gráficos
  pdf("E:/Oscar/DataScience/Kryterion/Trabajo final/Analisis_Variables.pdf", width = 10, height = 7)
  
  # 3. Definir un umbral para decidir si una variable numérica es discreta o continua
  # Si una variable numérica tiene MENOS de este número de valores únicos, haremos un diagrama de barras.
  # Si tiene más, haremos un histograma. Puedes ajustar este valor.
  umbral_valores_unicos <- 5

  
  # 4. Bucle para recorrer cada columna y generar un gráfico
  # Itera a través de los nombres de las columnas de tu data frame
  for (nombre_columna in names(input_data)) {

    # Extraemos solo la columna que nos interesa y la ponemos en un df simple.
    # La columna dentro de este df temporal se llamará 'datos_columna'.
    plot_df <- data.frame(datos_columna = input_data[[nombre_columna]])
 
    # Ahora construimos el ggplot usando este df temporal y un nombre de columna fijo ('datos_columna')
    p <- ggplot(plot_df, aes(x = datos_columna)) +
      labs(title = paste("Distribución de", nombre_columna), x = nombre_columna, y = "Frecuencia") +
      theme_minimal()
    
    #print(nombre_columna)
    # --- Lógica para decidir el tipo de gráfico ---
    
    # Si la columna es de tipo texto (character) o factor...

    if (is.character(plot_df[,1]) || is.factor(plot_df[,1])) {
      
      # ...hacemos un diagrama de barras.
      p <- p + geom_bar(fill = "steelblue", alpha = 0.8)
      print('debuger1')
      # Si la columna es de tipo numérico...
    } else if (is.numeric(plot_df[,1])) {
      
      # ...comprobamos cuántos valores únicos tiene (ignorando los NAs).
      n_unicos <- length(unique(na.omit(plot_df[,1])))

      print('debuger2')
      # Si tiene menos valores únicos que nuestro umbral...
      if (n_unicos < umbral_valores_unicos) {
        # ...la tratamos como categórica y hacemos un diagrama de barras.
        # Es importante convertirla a factor para que ggplot la trate como tal.
        p <- ggplot(plot_df, aes(x = factor(.data))) +
          labs(title = paste("Distribución de", nombre_columna), x = nombre_columna, y = "Frecuencia") +
          theme_minimal() +
          geom_bar(fill = "skyblue", alpha = 0.8)
        print('debuger3')
        # Si tiene muchos valores únicos...
      } else {
        # ...la tratamos como continua y hacemos un histograma.
        p <- p + geom_histogram(bins = 30, fill = "salmon", alpha = 0.8, color="white")
        print('debuger4')
      }
      
    } else {
      # Si la columna no es ni numérica ni de texto/factor, la saltamos.
      next
    }
    
    # Imprimimos el gráfico. Dentro de un bucle, es necesario usar print().
    print(p)

  }
  # Cierra el dispositivo PDF para finalizar y guardar el archivo
  dev.off()
}


# Función para realizar la imputación de mazo dirigida por tiempo
# Permite introducir como vector tanto los años que serán donantes como los años que serán receptores
imputar_con_tendencia <- function(valores, year, años_donante, años_receptor) {
  
  # 1. Identificar el "mazo" de donantes: valores de 2018 y 2019 que no sean NA
  mazo_donantes <- valores[year %in% años_donante & !is.na(valores)]
  
  # 2. Comprobación de seguridad: si no hay donantes para este estado, no hacemos nada
  if (length(mazo_donantes) == 0) {
    # Devolvemos los valores originales sin cambios para este grupo
    return(valores)
  }
  
  # 3. Identificar los índices de los valores a imputar (receptores)
  indices_a_imputar <- which(is.na(valores) & year %in% años_receptor)
  
  # 4. Si hay valores para imputar, los reemplazamos
  if (length(indices_a_imputar) > 0) {
    # Para cada NA, sacamos una muestra aleatoria (con reemplazo) del mazo de donantes
    valores_imputados <- sample(mazo_donantes, size = length(indices_a_imputar), replace = TRUE)
    
    # Reemplazamos los NAs con los nuevos valores
    valores[indices_a_imputar] <- valores_imputados
  }
  
  # 5. Devolver el vector completo con los valores ya imputados
  return(valores)
}
  
  #' Imputación por Regresión Lineal Temporal por Grupo
  #'
  #' Imputa valores NA en un vector numérico ('precios') ajustando un modelo
  #' de regresión lineal (precios ~ anios) con los datos observados y
  #' prediciendo los valores para los NAs.
  #'
  #' @param precios Vector numérico con los precios (puede contener NAs).
  #' @param anios Vector numérico con los años correspondientes a cada precio.
  #'
  #' @return Un vector numérico con los NAs imputados.
  
imputar_regresion <- function(valores, anios) {
    
    # 1. Crear un data frame temporal para trabajar más fácil
    df_temporal <- data.frame(estimacion = valores, anio = anios)
    
    # 2. Separar los datos en:
    #    - 'entrenamiento': filas con precios observados (para construir el modelo)
    #    - 'a_predecir': filas con precios faltantes (para usar el modelo)
    entrenamiento <- df_temporal[!is.na(df_temporal$estimacion), ]
    a_predecir <- df_temporal[is.na(df_temporal$estimacion), ]
    
    # 3. Comprobación de seguridad: si no hay datos para entrenar o nada que predecir,
    #    o si no hay suficientes datos para una regresión (necesitamos al menos 2 puntos),
    #    devolvemos los datos originales sin cambios para este grupo.
    if (nrow(entrenamiento) < 2 || nrow(a_predecir) == 0) {
      return(valores)
    }
    
    # 4. Ajustar el modelo de regresión lineal: precio en función del año
    modelo_lineal <- lm(estimacion ~ anio, data = entrenamiento)
    
    # 5. Predecir los precios para los años faltantes
    valores_predichos <- predict(modelo_lineal, newdata = a_predecir)
    
    # 6. Rellenar los huecos (NAs) en el vector original con las predicciones
    valores[is.na(valores)] <- valores_predichos
    
    # 7. Devolver el vector completo y ya imputado
    return(valores)
}


#Version mejorada y más flexible de la función para imputar por regresión
#' Imputación por Regresión Lineal usando años específicos para el modelo
#'
#' @param valores Vector numérico con los datos a imputar (contiene NAs).
#' @param anios Vector numérico con los años de cada observación.
#' @param anios_modelo Vector numérico con los años que se usarán para entrenar el modelo de regresión.

imputar_regresion_filtrada <- function(valores, anios, anios_modelo) {
  
  # 1. Crear un data frame temporal
  df_temporal <- data.frame(valor = valores, anio = anios)
  
  # 2. Definir el conjunto de entrenamiento usando SOLO los años especificados
  entrenamiento <- df_temporal[!is.na(df_temporal$valor) & df_temporal$anio %in% anios_modelo, ]
  
  # 3. Identificar las filas que necesitan ser predichas (todas las que tienen NA)
  a_predecir <- df_temporal[is.na(df_temporal$valor), ]
  
  # 4. Comprobación de seguridad
  if (nrow(entrenamiento) < 2 || nrow(a_predecir) == 0) {
    return(valores)
  }
  
  # 5. Ajustar el modelo de regresión lineal
  modelo_lineal <- lm(valor ~ anio, data = entrenamiento)
  
  # 6. Predecir los precios para los años faltantes
  valores_predichos <- predict(modelo_lineal, newdata = a_predecir)
  
  # 7. Rellenar los NAs
  valores[is.na(valores)] <- valores_predichos
  
  # 8. Devolver el vector imputado
  return(valores)
}

#' Imputación Híbrida: Regresión con Fallback a Media/Mediana
#'
#' Imputa valores NA. Si hay suficientes datos (>=2), usa regresión lineal (valor ~ anio).
#' Si hay pocos datos (>0 pero <2), usa la mediana de los valores disponibles como fallback.
#'
#' @param valores Vector numérico con los datos a imputar.
#' @param anios Vector numérico con los años correspondientes.
#'
#' @return Un vector numérico con los NAs imputados.

imputar_hibrido_regresion <- function(valores, anios) {
  
  # 1. Crear un data frame temporal
  df_temporal <- data.frame(estimacion = valores, anio = anios)
  
  # 2. Separar datos de entrenamiento y datos a predecir
  entrenamiento <- df_temporal[!is.na(df_temporal$estimacion), ]
  a_predecir <- df_temporal[is.na(df_temporal$estimacion), ]
  
  # 3. Comprobación inicial: si no hay nada que imputar, salimos.
  if (nrow(a_predecir) == 0) {
    return(valores)
  }
  
  # --- LÓGICA HÍBRIDA MEJORADA ---
  
  # CASO 1: Hay suficientes datos para una regresión
  if (nrow(entrenamiento) >= 2) {
    
    # Mensaje para saber qué método se está usando (útil para depurar)
    # message("Usando regresión lineal...")
    
    modelo_lineal <- lm(estimacion ~ anio, data = entrenamiento)
    valores_predichos <- predict(modelo_lineal, newdata = a_predecir)
    valores[is.na(valores)] <- valores_predichos
    return(valores)
    
    # CASO 2 (FALLBACK): Hay algún dato, pero no suficientes para regresión (es decir, solo 1)
  } else if (nrow(entrenamiento) > 0) {
    
    # message("Insuficientes datos para regresión. Usando fallback a la mediana.")
    
    # Calculamos la mediana de los pocos datos que tenemos (si solo hay uno, será ese mismo valor)
    valor_de_relleno <- median(entrenamiento$estimacion, na.rm = TRUE)
    
    # Rellenamos todos los NAs con ese único valor
    valores[is.na(valores)] <- valor_de_relleno
    return(valores)
    
    # CASO 3: No hay ningún dato de entrenamiento
  } else {
    # No podemos hacer nada, devolvemos los datos tal como estaban
    # message("No hay datos disponibles para imputar.")
    return(valores)
  }
}

#Función para crear un pdf con la representación gráfica de los boxplots de las distintas variables.

print_boxplots <- function(input_data, ruta_completa_pdf) {
  
  # Cargar las librerías necesarias
  library(ggplot2)
  library(dplyr)
  
  # 1. Identificar solo las columnas que son numéricas
  columnas_numericas <- names(input_data)[sapply(input_data, is.numeric)]
  
  # Mensaje informativo
  message(paste("Se generarán boxplots para", length(columnas_numericas), "variables numéricas."))
  
  # 2. Abrir el dispositivo PDF para guardar los gráficos
  pdf(ruta_completa_pdf, width = 8, height = 6)
  
  # 3. Bucle para recorrer cada columna numérica
  for (nombre_columna in columnas_numericas) {
    
    # Mensaje para ver el progreso en la consola
    message(paste("Creando boxplot para:", nombre_columna))
    
    # Creamos el gráfico usando tryCatch para evitar que un error detenga todo
    tryCatch({
      
      p <- ggplot(input_data, aes(x = "", y = .data[[nombre_columna]])) + # <- La clave está aquí
        geom_boxplot(
          fill = "lightseagreen", # Color de relleno
          alpha = 0.7,
          outlier.color = "red",  # Resaltar los outliers en rojo
          outlier.shape = 18,     # Cambiar la forma del punto del outlier
          outlier.size = 2        # Hacer los outliers un poco más grandes
        ) +
        labs(
          title = paste("Boxplot de", nombre_columna),
          subtitle = "Para la detección de outliers",
          x = "",                  # No necesitamos etiqueta en el eje X
          y = nombre_columna
        ) +
        theme_minimal()
      
      # Imprimimos el gráfico en el PDF
      print(p)
      
    }, error = function(e) {
      # Si ocurre un error, imprime un mensaje y continúa
      message(paste("   -> ERROR al procesar '", nombre_columna, "'. Error:", e$message))
    }) # Fin de tryCatch
    
  } # Fin del bucle for
  
  # 4. Cierra el dispositivo PDF para finalizar y guardar el archivo
  dev.off()
  
  message(paste("Proceso finalizado. El archivo PDF ha sido creado en:", ruta_completa_pdf))
}

linear_regression_assumptions <- function(regression_model) {
  
  # --- Forzar la impresión de cada test ---
  
  cat("\n========================================================\n")
  cat("       INICIO DE LOS DIAGNÓSTICOS DEL MODELO\n")
  cat("========================================================\n\n")
  
  # 1. Especificación (Test RESET)
  cat("--- (1) Test RESET de Ramsey para Especificación del Modelo ---\n\n")
  print(resettest(regression_model))
  cat("\n--------------------------------------------------------\n")
  
  # 2. Normalidad de los Residuos (Test de Shapiro-Wilk)
  cat("--- (2) Test de Shapiro-Wilk para Normalidad de Residuos ---\n\n")
  print(shapiro.test(regression_model$residuals))
  cat("\n--------------------------------------------------------\n")
  
  # 3. Media Cero de los Errores (Test T)
  cat("--- (3) Test T para Media de los Residuos igual a Cero ---\n\n")
  print(t.test(regression_model$residuals, mu = 0))
  cat("\n--------------------------------------------------------\n")
  
  # 4. Homocedasticidad (Test de Breusch-Pagan)
  cat("--- (4) Test de Breusch-Pagan para Homocedasticidad ---\n\n")
  print(bptest(regression_model))
  cat("\n--------------------------------------------------------\n")
  
  # 5. Independencia de los Errores (Test de Durbin-Watson)
  cat("--- (5) Test de Durbin-Watson para Autocorrelación ---\n\n")
  print(durbinWatsonTest(regression_model))
  
  cat("\n========================================================\n")
  cat("        FIN DE LOS DIAGNÓSTICOS DEL MODELO\n")
  cat("========================================================\n\n")
  
  # --- Generar los gráficos ---
  # RStudio los mostrará en la pestaña "Plots"
  
  par(mfrow = c(2, 2)) # Prepara una cuadrícula de 2x2 para los gráficos
  plot(regression_model)
  par(mfrow = c(1, 1)) # Restaura la configuración a un solo gráfico
}

plot_confidence_intervals <- function(regression_model,dataset){
  #Dibujo intervalos de confianza y predicción.
  # 2. Calcular ambos intervalos (confianza y predicción) al 95%
  intervalo_confianza <- predict(regression_model, interval = "confidence", level = 0.95)
  colnames(intervalo_confianza)[1:3] <- c("fit","conf_lwr","conf_upr")
  
  intervalo_prediccion <- predict(regression_model, interval = "prediction", level = 0.95)
  intervalo_prediccion <- intervalo_prediccion[,-1]
  colnames(intervalo_prediccion)[1:2] <- c("pred_lwr","pred_upr")
  
  # 3. Añadir estos intervalos a nuestro data frame original
  total_data_con_intervalos <- cbind(dataset, intervalo_confianza, intervalo_prediccion)
  
  
  # Vemos cómo ha quedado el data frame
  head(total_data_con_intervalos)
  
  #Creamos el gráfico con ggplot
  ggplot(total_data_con_intervalos, aes(x = PC1, y = dataset[,1])) +
    
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
      y = "Variable Objetivo "
    ) +
    theme_minimal()
}