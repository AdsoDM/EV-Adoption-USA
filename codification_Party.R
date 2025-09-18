# 3. Convertir la variable categórica 'Party' en dummy (0s y 1s)
data_para_similitud <- dummy_cols(data_para_similitud, 
                                  select_columns = "Party", 
                                  remove_first_dummy = TRUE, # Para evitar multicolinealidad
                                  remove_selected_columns = TRUE)