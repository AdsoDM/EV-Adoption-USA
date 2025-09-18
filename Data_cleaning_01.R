#Final Practice_v01_EV Adoption USA

#Load libraries
#Data Cleaning
library(caTools) #necessary to create the split for separating data into train and test
library(dplyr) # included in tidyverse. loading tidyverse would give us access to all necessary tools
library(fastDummies) #necessary to create Dummy variables to convert categorical variables to numbers
library(ggplot2)
library(dlookr) #necessary for normality tests
library(VIM) #necessary for hot-deck imputation
library(FNN) #necessary for KNN imputation

#Load my functions
source(".../adhoc_functions.R")

#Load data
ev_adoption <- read.csv(".../EV_data.csv")
#Remove the first column as it is duplicated and rename the second to index

ev_adpotion
View(ev_adoption)

#Data Wrangling
#The table complies with Codd's 3rd normal form and Data Wrangling is not necessary
#Codd's 3rd normal form requires that:
# (1) Each variable forms a column
# (2) Each observation forms a row
# (3) Each type of observational unit forms a table

#Remove the first column as it is duplicated and rename the second to index
ev_adoption_01 <- ev_adoption[,-1]
colnames(ev_adoption_01)[1]<- "Index"


#Data cleaning
ev_adoption_01_BU <- ev_adoption_01 #Make a backup copy of the dataframe
str(ev_adoption_01)

#Analyze missing values

ina <- is.na(unlist(ev_adoption_01)) #creates a vector with the 'not available' values of the dataset
sum(ina)

#Check how many instances are complete (have no missing data)
sum(complete.cases(ev_adoption_01))

#It makes sense to split the data from 50/50 to 80/20 (training/test)

set.seed(123) #until the seed is changed, this seed will be used for any pseudo-randomization command used hereafter

#Create the split to separate data into training and test
ksplit=sample.split(ev_adoption_01$Index, SplitRatio = 0.80)

training_set = subset(ev_adoption_01, ksplit==TRUE)
test_set = subset(ev_adoption_01, ksplit==FALSE)

#Handling missing data

print_atributes(ev_adoption_01)

#Imputation analysis

# 1. Create an empty vector to store the results
conteo_nas_por_columna <- c()

#Loop to get the missing values per variable
for (nombre_columna in names(ev_adoption_01)) {
  # 3. For each column, count the NAs and save the result
  #    is.na() creates a TRUE/FALSE vector. sum() counts TRUE as 1 and FALSE as 0.
  conteo_na <- sum(is.na(ev_adoption_01[[nombre_columna]]))
  
  # 4. Add the result to the results vector
  conteo_nas_por_columna[nombre_columna] <- conteo_na
}

#Variable Fuel economy

#plot the evolution of the 'fuel_economy' variable
#The 'fuel_economy' variable is the average consumption for the entire country (all 51 states each year). Two full years of data are missing, 2016 and 2017.
ggplot(data = ev_adoption_01, aes(x=Index, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Index", y="Average vehicle consumption in the state")
ggplot(data = ev_adoption_01, aes(x=year, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Year", y="Average vehicle consumption in the state")

#Analysis of the fuel_economy variable
#plot the evolution of the population by date
#Count the repetitions of the different values
ev_adoption_01 %>%  count(fuel_economy, sort = TRUE)

#plot the evolution of the 'fuel_economy' variable
ggplot(data = ev_adoption_01, aes(x=year, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Year", y="Average fuel consumption of vehicles in the USA")
ggplot(data = ev_adoption_01, aes(x=Index, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Index", y="Average fuel consumption of vehicles in the USA")
ggplot(data = ev_adoption_01, aes(x=fuel_economy))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Average fuel consumption of vehicles in the USA", x="Sensitivity", y="Frequency") +
  theme_minimal()

#Check for normality
ev_adoption_01 %>% plot_normality(fuel_economy) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_01$fuel_economy)
print(resultado_shapiro)

#I will impute using linear regression
# Apply the regression imputation function to each state
# I assume the price column is named 'gasoline_price_per_gallon'
#I will perform the regression with the values from 2018 to 2020


ev_adoption_01 <- ev_adoption_01 %>%
  # We use mutate to replace the price column with the imputed version
  mutate(
    fuel_economy = imputar_regresion_filtrada(
      valores = fuel_economy,
      anios = year,
      anios_modelo = c(2018,2019,2020)
    )
  )

# Verify the result for a state and the years of interest
# You should see that the values for 2016 and 2017 are now filled
# with values that follow the trend of the subsequent years.
ev_adoption_01 %>%
  filter(state == "California", year %in% 2016:2019) %>%
  select(state, year, fuel_economy)

#plot the evolution of the 'fuel_economy' variable
ggplot(data = ev_adoption_01, aes(x=year, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Year", y="Average fuel consumption of vehicles in the USA")
ggplot(data = ev_adoption_01, aes(x=Index, y = fuel_economy))+ geom_line(aes(color = fuel_economy))+labs(x="Index", y="Average fuel consumption of vehicles in the USA")
ggplot(data = ev_adoption_01, aes(x=fuel_economy))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Average fuel consumption of vehicles in the USA", x="Sensitivity", y="Frequency") +
  theme_minimal()



#Analysis of the Incentives variable
ggplot(data = ev_adoption_01, aes(x=year, y = Incentives))+ geom_line(aes(color = Incentives))+labs(x="Year", y="Presence of incentives from the State")
ggplot(data = ev_adoption_01, aes(x=Incentives))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Distribution of electric vehicle incentives", x="Number of incentives", y="Frequency") +
  theme_minimal()

#I will try to impute by mean
ev_adoption_02 <- ev_adoption_01
ev_adoption_02$Incentives = ifelse(is.na(ev_adoption_01$Incentives), ave(ev_adoption_01$Incentives, FUN = function(x) mean(x, na.rm = TRUE)), ev_adoption_01$Incentives)

#Analysis of the Number.of.Metro.Organizing.Committees variable
ggplot(data = ev_adoption_01, aes(x=year, y = Number.of.Metro.Organizing.Committees))+ geom_line(aes(color = Number.of.Metro.Organizing.Committees))+labs(x="Year", y="Number of metropolitan organizing committees")
ggplot(data = ev_adoption_01, aes(x=Index, y = Number.of.Metro.Organizing.Committees))+ geom_line(aes(color = Number.of.Metro.Organizing.Committees))+labs(x="Index", y="Number of metropolitan organizing committees")
ggplot(data = ev_adoption_01, aes(x=Number.of.Metro.Organizing.Committees))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Number of metropolitan organizing committees", x="Number of incentives", y="Frequency") +
  theme_minimal()

#Data for one year is missing. I will impute as the mean of the previous and next year for each state.

# Imputation in a single step
ev_adoption_03 <- ev_adoption_02 %>%
  
  # 1. Group by state so that calculations are done separately for each one
  group_by(state) %>%
  # 2. Use mutate to create the mean and then use it to impute
  mutate(
    # FIRST: We calculate the mean for 2018, 2020 and save it in a temporary column.
    media_para_imputar = mean(Number.of.Metro.Organizing.Committees[year %in% c(2018, 2020)], na.rm = TRUE),
    # SECOND: We modify the original column using the temporary column we just created.
    Number.of.Metro.Organizing.Committees = ifelse(
      # Condition: If the year is 2019 and the value is NA...
      year == 2019 & is.na(Number.of.Metro.Organizing.Committees),
      # ...then, use the mean we calculated in the previous step.
      round(media_para_imputar),
      # ...otherwise, leave the original value.
      Number.of.Metro.Organizing.Committees
    )
  ) %>%
  
  # 3. Ungroup, a good practice after group_by and mutate
  ungroup() %>%
  
  # 4. Remove the temporary column that we no longer need
  select(-media_para_imputar)

# Verify the result (it should be identical to the previous method's)
ev_adoption_03 %>%
  filter(state == "Alabama", year %in% c(2018, 2019, 2020)) %>%
  select(state, year, Number.of.Metro.Organizing.Committees)
    
##Analysis of the affectweather variable
ggplot(data = ev_adoption_01, aes(x=year, y = affectweather))+ geom_line(aes(color = affectweather))+labs(x="Year", y="Measure of sensitivity or concern about climate change")
ggplot(data = ev_adoption_01, aes(x=Index, y = affectweather))+ geom_line(aes(color = affectweather))+labs(x="Index", y="Measure of sensitivity or concern about climate change")
ggplot(data = ev_adoption_01, aes(x=affectweather))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of sensitivity or concern about climate change", x="Sensitivity", y="Frequency") +
  theme_minimal()

ev_adoption_03 %>% plot_normality(affectweather) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_03$affectweather)
print(resultado_shapiro)
#I will try to impute by mean
ev_adoption_04 <- ev_adoption_03
#I tried imputing by mean and I don't like the result. The variable has lost its normal properties
ev_adoption_04$affectweather = ifelse(is.na(ev_adoption_04$affectweather), ave(ev_adoption_04$affectweather, FUN = function(x) mean(x, na.rm = TRUE)), ev_adoption_04$affectweather)

#I will try hot-deck imputation
set.seed(123) # Again, for reproducibility

# Perform stratified imputation by state ('state')
ev_adoption_04 <- hotdeck(
  ev_adoption_04,
  variable = "affectweather",
  domain_var = "state" # The variable to create the "strata" or groups
)

# You can even stratify by more than one variable
# set.seed(123)
# ev_adoption_imputado_estratificado_2 <- hotdeck(
#   ev_adoption_01,
#   variable = "affectweather",
#   domain_var = c("state", "Party") # Use donors from the same state and party
# )

#Check for normality
ev_adoption_04 %>% plot_normality(affectweather) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_04$affectweather)
print(resultado_shapiro)

#The normality has slightly worsened but it is still an acceptable hypothesis
#We have already gone from 1505 missing data to 1344
ina <- is.na(unlist(ev_adoption_04)) #creates a vector with the 'not available' values of the dataset
sum(ina)

#Variable DEVHarm
ggplot(data = ev_adoption_04, aes(x=year, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Year", y="Measure of sensitivity or concern about the impact of climate change on economic growth")
ggplot(data = ev_adoption_04, aes(x=Index, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Index", y="Measure of sensitivity or concern about the impact of climate change on economic growth")
ggplot(data = ev_adoption_04, aes(x=devharm))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of sensitivity or concern about climate change", x="Sensitivity", y="Frequency") +
  theme_minimal()

#Check for normality
ev_adoption_04 %>% plot_normality(devharm) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_04$devharm)
print(resultado_shapiro)

#I will try hot-deck imputation
set.seed(123) # Again, for reproducibility

#The data has a temporal trend.
#Concern is increasing over the years. Therefore, I will implement a hot-deck imputation but taking values only from the years 2018 and 2019
# Apply the imputation function to each state
ev_adoption_05 <- ev_adoption_04 %>%
  group_by(state) %>%  # We group by state so that the function is applied to each one separately
  mutate(
    devharm = imputar_con_tendencia(devharm, year, c(2018,2019),c(2016,2017)) # We apply our function to the devharm and year columns
  ) %>%
  ungroup() # We ungroup as a good practice

# Verify the result for a state and the years of interest
ev_adoption_05 %>%
  filter(state == "Alabama", year %in% 2016:2019) %>%
  select(state, year, devharm)

#Check normality and plots
ggplot(data = ev_adoption_05, aes(x=year, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Year", y="Measure of sensitivity or concern about the impact of climate change on economic growth")
ggplot(data = ev_adoption_05, aes(x=Index, y = devharm))+ geom_line(aes(color = devharm))+labs(x="Index", y="Measure of sensitivity or concern about the impact of climate change on economic growth")
ggplot(data = ev_adoption_05, aes(x=devharm))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of sensitivity or concern about climate change", x="Sensitivity", y="Frequency") +
  theme_minimal()
ev_adoption_05 %>% plot_normality(devharm) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_05$devharm)
print(resultado_shapiro)
#Normality is lost but a temporal trend is already visible and therefore the data is not normal to begin with

#Variable Discuss

#Variable Discuss
ggplot(data = ev_adoption_05, aes(x=state, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Year", y="Measure of the frequency with which people discuss environmental issues")
ggplot(data = ev_adoption_05, aes(x=Index, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Index", y="Measure of the frequency with which people discuss environmental issues")
ggplot(data = ev_adoption_05, aes(x=devharm))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of the frequency with which people discuss environmental issues", x="Sensitivity", y="Frequency") +
  theme_minimal()
ev_adoption_05 %>% plot_normality(discuss) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_05$discuss)
print(resultado_shapiro)

# Perform stratified imputation by state ('state')
ev_adoption_06 <- hotdeck(
  ev_adoption_05,
  variable = "discuss",
  domain_var = "state" # The variable to create the "strata" or groups
)

#Check normality and plots
ggplot(data = ev_adoption_06, aes(x=state, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Year", y="Measure of the frequency with which people discuss environmental issues")
ggplot(data = ev_adoption_06, aes(x=Index, y = discuss))+ geom_line(aes(color = discuss))+labs(x="Index", y="Measure of the frequency with which people discuss environmental issues")
ggplot(data = ev_adoption_06, aes(x=discuss))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of the frequency with which people discuss environmental issues", x="Sensitivity", y="Frequency") +
  theme_minimal()
ev_adoption_06 %>% plot_normality(discuss) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_06$discuss)
print(resultado_shapiro)

#Variable exp
ggplot(data = ev_adoption_06, aes(x=state, y = exp))+ geom_line(aes(color = exp))+labs(x="state", y="Measure of experience and knowledge on environmental issues")
ggplot(data = ev_adoption_06, aes(x=year, y = exp))+ geom_line(aes(color = exp))+labs(x="Year", y="Measure of experience and knowledge on environmental issues")
ggplot(data = ev_adoption_06, aes(x=Index, y = exp))+ geom_line(aes(color = exp))+labs(x="Index", y="Measure of experience and knowledge on environmental issues")
ggplot(data = ev_adoption_06, aes(x=exp))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of experience and knowledge on environmental issues", x="Knowledge", y="Frequency") +
  theme_minimal()
ev_adoption_06 %>% plot_normality(exp) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_06$exp)
print(resultado_shapiro)

## Perform stratified imputation by state ('state')
ev_adoption_07 <- hotdeck(
  ev_adoption_06,
  variable = "exp",
  domain_var = "state" # The variable to create the "strata" or groups
)

#Check normality and plot
ggplot(data = ev_adoption_07, aes(x=state, y = exp))+ geom_line(aes(color = exp))+labs(x="state", y="Measure of experience and knowledge on environmental issues")
ggplot(data = ev_adoption_07, aes(x=year, y = exp))+ geom_line(aes(color = exp))+labs(x="Year", y="Measure of experience and knowledge on environmental issues")
ggplot(data = ev_adoption_07, aes(x=Index, y = exp))+ geom_line(aes(color = exp))+labs(x="Index", y="Measure of experience and knowledge on environmental issues")
ggplot(data = ev_adoption_07, aes(x=exp))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of experience and knowledge on environmental issues", x="Knowledge", y="Frequency") +
  theme_minimal()
ev_adoption_07 %>% plot_normality(exp) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_07$exp)
print(resultado_shapiro)

#Variable localofficials
ggplot(data = ev_adoption_07, aes(x=state, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="state", y="Measure of trust in local environmental offices")
ggplot(data = ev_adoption_07, aes(x=year, y = localofficials))+ geom_line(aes(color = exp))+labs(x="Year", y="Measure of trust in local environmental offices")
ggplot(data = ev_adoption_07, aes(x=Index, y = localofficials))+ geom_line(aes(color = exp))+labs(x="Index", y="Measure of trust in local environmental offices")
ggplot(data = ev_adoption_07, aes(x=localofficials))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of trust in local environmental offices", x="Knowledge", y="Frequency") +
  theme_minimal()
ev_adoption_07 %>% plot_normality(localofficials) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_07$localofficials)
print(resultado_shapiro)


## Perform stratified imputation by state ('state')
ev_adoption_08 <- hotdeck(
  ev_adoption_07,
  variable = "localofficials",
  domain_var = "state" # The variable to create the "strata" or groups
)

#Check normality and plot
ggplot(data = ev_adoption_08, aes(x=state, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="state", y="Measure of trust in local environmental offices")
ggplot(data = ev_adoption_08, aes(x=year, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="Year", y="Measure of trust in local environmental offices")
ggplot(data = ev_adoption_08, aes(x=Index, y = localofficials))+ geom_line(aes(color = localofficials))+labs(x="Index", y="Measure of trust in local environmental offices")
ggplot(data = ev_adoption_08, aes(x=localofficials))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of trust in local environmental offices", x="Knowledge", y="Frequency") +
  theme_minimal()
ev_adoption_08 %>% plot_normality(localofficials) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_08$localofficials)
print(resultado_shapiro)

#Variable personal
ggplot(data = ev_adoption_08, aes(x=state, y = personal))+ geom_line(aes(color = personal))+labs(x="state", y="Measure of personal responsibility towards the environment")
ggplot(data = ev_adoption_08, aes(x=year, y = personal))+ geom_line(aes(color = personal))+labs(x="Year", y="Measure of personal responsibility towards the environment")
ggplot(data = ev_adoption_08, aes(x=Index, y = personal))+ geom_line(aes(color = personal))+labs(x="Index", y="Measure of personal responsibility towards the environment")
ggplot(data = ev_adoption_08, aes(x=personal))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of personal responsibility towards the environment", x="Responsibility", y="Frequency") +
  theme_minimal()
ev_adoption_08 %>% plot_normality(personal) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_08$personal)
print(resultado_shapiro)

## Perform stratified imputation by state ('state')
ev_adoption_09 <- hotdeck(
  ev_adoption_08,
  variable = "personal",
  domain_var = "state" # The variable to create the "strata" or groups
)

#Check normality and plot
ggplot(data = ev_adoption_09, aes(x=state, y = personal))+ geom_line(aes(color = personal))+labs(x="state", y="Measure of personal responsibility towards the environment")
ggplot(data = ev_adoption_09, aes(x=year, y = personal))+ geom_line(aes(color = personal))+labs(x="Year", y="Measure of personal responsibility towards the environment")
ggplot(data = ev_adoption_09, aes(x=Index, y = personal))+ geom_line(aes(color = personal))+labs(x="Index", y="Measure of personal responsibility towards the environment")
ggplot(data = ev_adoption_09, aes(x=personal))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of personal responsibility towards the environment", x="Responsibility", y="Frequency") +
  theme_minimal()
ev_adoption_09 %>% plot_normality(personal) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_09$personal)
print(resultado_shapiro)

#Variable reducetax
ggplot(data = ev_adoption_09, aes(x=state, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="state", y="Measure of support for reducing taxes with environmental policies")
ggplot(data = ev_adoption_09, aes(x=year, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Year", y="Measure of support for reducing taxes with environmental policies")
ggplot(data = ev_adoption_09, aes(x=Index, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Index", y="Measure of support for reducing taxes with environmental policies")
ggplot(data = ev_adoption_09, aes(x=reducetax))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of support for reducing taxes with environmental policies", x="Tax reduction", y="Frequency") +
  theme_minimal()
ev_adoption_09 %>% plot_normality(reducetax) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_09$reducetax)
print(resultado_shapiro)

## Perform stratified imputation by state ('state')
ev_adoption_10 <- hotdeck(
  ev_adoption_09,
  variable = "reducetax",
  domain_var = "state" # The variable to create the "strata" or groups
)

#Check normality and plot
ggplot(data = ev_adoption_10, aes(x=state, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="state", y="Measure of support for reducing taxes with environmental policies")
ggplot(data = ev_adoption_10, aes(x=year, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Year", y="Measure of support for reducing taxes with environmental policies")
ggplot(data = ev_adoption_10, aes(x=Index, y = reducetax))+ geom_line(aes(color = reducetax))+labs(x="Index", y="Measure of support for reducing taxes with environmental policies")
ggplot(data = ev_adoption_10, aes(x=reducetax))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of support for reducing taxes with environmental policies", x="Tax reduction", y="Frequency") +
  theme_minimal()
ev_adoption_10 %>% plot_normality(reducetax) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_10$reducetax)
print(resultado_shapiro)

#Variable regulate
ggplot(data = ev_adoption_10, aes(x=state, y = regulate))+ geom_line(aes(color = regulate))+labs(x="state", y="Measure of government support for environmental regulations")
ggplot(data = ev_adoption_10, aes(x=year, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Year", y="Measure of government support for environmental regulations")
ggplot(data = ev_adoption_10, aes(x=Index, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Index", y="Measure of government support for environmental regulations")
ggplot(data = ev_adoption_10, aes(x=regulate))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of government support for environmental regulations", x="Tax reduction", y="Frequency") +
  theme_minimal()
ev_adoption_10 %>% plot_normality(regulate) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_10$regulate)
print(resultado_shapiro)

#Use hot-deck imputation segregating by state and specific years
#Concern is increasing over the years. Therefore, I will implement a hot-deck imputation but taking values only from the years 2018 and 2019
# Apply the imputation function to each state
ev_adoption_11 <- ev_adoption_10 %>%
  group_by(state) %>%  # We group by state so that the function is applied to each one separately
  mutate(
    regulate = imputar_con_tendencia(regulate, year, c(2018,2019,2020), c(2016,2017)) # We apply our function to the devharm and year columns
  ) %>%
  ungroup() # We ungroup as a good practice

# Verify the result for a state and the years of interest
ev_adoption_11 %>%
  filter(state == "Alabama", year %in% 2016:2019) %>%
  select(state, year, regulate)

#Check normality and plot
ggplot(data = ev_adoption_11, aes(x=state, y = regulate))+ geom_line(aes(color = regulate))+labs(x="state", y="Measure of government support for environmental regulations")
ggplot(data = ev_adoption_11, aes(x=year, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Year", y="Measure of government support for environmental regulations")
ggplot(data = ev_adoption_11, aes(x=Index, y = regulate))+ geom_line(aes(color = regulate))+labs(x="Index", y="Measure of government support for environmental regulations")
ggplot(data = ev_adoption_11, aes(x=regulate))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of government support for environmental regulations", x="Government support", y="Frequency") +
  theme_minimal()
ev_adoption_11 %>% plot_normality(regulate) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_11$regulate)
print(resultado_shapiro)

#Variable worried
ggplot(data = ev_adoption_11, aes(x=state, y = worried))+ geom_line(aes(color = worried))+labs(x="state", y="Measure of concern about environmental problems")
ggplot(data = ev_adoption_11, aes(x=year, y = worried))+ geom_line(aes(color = worried))+labs(x="Year", y="Measure of concern about environmental problems")
ggplot(data = ev_adoption_11, aes(x=Index, y = worried))+ geom_line(aes(color = worried))+labs(x="Index", y="Measure of concern about environmental problems")
ggplot(data = ev_adoption_11, aes(x=worried))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of concern about environmental problems", x="Concern", y="Frequency") +
  theme_minimal()
ev_adoption_11 %>% plot_normality(worried) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_11$worried)
print(resultado_shapiro)

## Perform stratified imputation by state ('state')
ev_adoption_12 <- hotdeck(
  ev_adoption_11,
  variable = "worried",
  domain_var = "state" # The variable to create the "strata" or groups
)

#Check normality and plot
ggplot(data = ev_adoption_12, aes(x=state, y = worried))+ geom_line(aes(color = worried))+labs(x="state", y="Measure of concern about environmental problems")
ggplot(data = ev_adoption_12, aes(x=year, y = worried))+ geom_line(aes(color = worried))+labs(x="Year", y="Measure of concern about environmental problems")
ggplot(data = ev_adoption_12, aes(x=Index, y = worried))+ geom_line(aes(color = worried))+labs(x="Index", y="Measure of concern about environmental problems")
ggplot(data = ev_adoption_12, aes(x=worried))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Measure of concern about environmental problems", x="Concern", y="Frequency") +
  theme_minimal()
ev_adoption_12 %>% plot_normality(worried) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_12$worried)
print(resultado_shapiro)

#Variable gasoline_price_per_gallon
ggplot(data = ev_adoption_12, aes(x=state, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="state", y="Average gasoline price in the state")
ggplot(data = ev_adoption_12, aes(x=year, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Year", y="Average gasoline price in the state")
ggplot(data = ev_adoption_12, aes(x=Index, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Index", y="Average gasoline price in the state")
ggplot(data = ev_adoption_12, aes(x=gasoline_price_per_gallon))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Average gasoline price in the state", x="Gasoline price", y="Frequency") +
  theme_minimal()
ev_adoption_12 %>% plot_normality(gasoline_price_per_gallon) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_12$gasoline_price_per_gallon)
print(resultado_shapiro)

#I will impute using linear regression
# Apply the regression imputation function to each state
# I assume the price column is named 'gasoline_price_per_gallon'
ev_adoption_13 <- ev_adoption_12 %>%
  
  # We group by state so that the imputation is done independently for each one
  group_by(state) %>%
  
  # We use mutate to replace the price column with the imputed version
  mutate(
    gasoline_price_per_gallon = imputar_regresion(
      valores = gasoline_price_per_gallon,
      anios = year
    )
  ) %>%
  
  # We ungroup as a good practice
  ungroup()

# Verify the result for a state and the years of interest
# You should see that the values for 2016 and 2017 are now filled
# with values that follow the trend of the subsequent years.
ev_adoption_13 %>%
  filter(state == "California", year %in% 2016:2019) %>%
  select(state, year, gasoline_price_per_gallon)

#Check normality and plot
ggplot(data = ev_adoption_13, aes(x=state, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="state", y="Average gasoline price in the state")
ggplot(data = ev_adoption_13, aes(x=year, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Year", y="Average gasoline price in the state")
ggplot(data = ev_adoption_13, aes(x=Index, y = gasoline_price_per_gallon))+ geom_line(aes(color = gasoline_price_per_gallon))+labs(x="Index", y="Average gasoline price in the state")
ggplot(data = ev_adoption_13, aes(x=gasoline_price_per_gallon))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Average gasoline price in the state", x="Gasoline price", y="Frequency") +
  theme_minimal()
ev_adoption_13 %>% plot_normality(gasoline_price_per_gallon) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_13$gasoline_price_per_gallon)
print(resultado_shapiro)

#Variable Trucks
ggplot(data = ev_adoption_13, aes(x=state, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="state", y="Number of trucks")
ggplot(data = ev_adoption_13, aes(x=year, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Year", y="Number of trucks")
ggplot(data = ev_adoption_13, aes(x=Index, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Index", y="Number of trucks")
ggplot(data = ev_adoption_13, aes(x=Trucks))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Number of trucks", x="Trucks", y="Frequency") +
  theme_minimal()
ev_adoption_13 %>% plot_normality(Trucks) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_13$Trucks)
print(resultado_shapiro)

#I will impute using linear regression
# Apply the regression imputation function to each state
# I assume the price column is named 'gasoline_price_per_gallon'
ev_adoption_14 <- ev_adoption_13 %>%
  
  # We group by state so that the imputation is done independently for each one
  group_by(state) %>%
  
  # We use mutate to replace the price column with the imputed version
  mutate(
    Trucks = imputar_regresion(
      valores = Trucks,
      anios = year
    )
  ) %>%
  
  # We ungroup as a good practice
  ungroup()

# Verify the result for a state and the years of interest
# You should see that the values for 2016 and 2017 are now filled
# with values that follow the trend of the subsequent years.
ev_adoption_14 %>%
  filter(state == "New York", year %in% 2016:2019) %>%
  select(state, year, Trucks)

#Check normality and plot
ggplot(data = ev_adoption_14, aes(x=state, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="state", y="Number of trucks")
ggplot(data = ev_adoption_14, aes(x=year, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Year", y="Number of trucks")
ggplot(data = ev_adoption_14, aes(x=Index, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Index", y="Number of trucks")
ggplot(data = ev_adoption_14, aes(x=Trucks))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Number of trucks", x="Trucks", y="Frequency") +
  theme_minimal()
ev_adoption_14 %>% plot_normality(Trucks) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_14$Trucks)
print(resultado_shapiro)

#The District of Columbia has no data, therefore it is not possible to impute them using regression.
#We must resort to the KNN (k-nearest neighbours) strategy

#Step 1: define key data and standardization

# 1. Prepare the data for the similarity analysis
#    We will use the data from the most recent and complete year (e.g., 2023) to define structural similarity.
data_para_similitud <- ev_adoption_14 %>%
  filter(year == 2023) %>%
  # Select the columns that define similarity
  select(state, Per_Cap_Income, Population_20_64, gasoline_price_per_gallon, Bachelor_Attainment, EV.Share....)


# 2. Standardize the numerical variables (CRITICAL Step!)
#    This ensures that no variable dominates the distance calculation just by having a larger scale.
#    We save the state names and then standardize the rest.
states_list <- data_para_similitud$state
data_estandarizada <- data.frame(scale(select(data_para_similitud, -state)))
row.names(data_estandarizada) <- states_list

#Step 2: Find the K nearest neighbours to the District of Columbia

# 3. Separate D.C.'s data from the rest
dc_data_std <- data_estandarizada["District Of Columbia", ]
otros_estados_std <- data_estandarizada[row.names(data_estandarizada) != "District Of Columbia", ]

# 4. Find the k nearest neighbours (we will use k=5)
#    get.knnx searches for the 'k' neighbours in 'otros_estados_std' for each point in 'dc_data_std'
k <- 5
vecinos <- get.knnx(data = otros_estados_std, 
                    query = dc_data_std, 
                    k = k)

# 5. Extract the names of the neighbouring states
nombres_vecinos <- row.names(otros_estados_std)[vecinos$nn.index]
print(paste("Los", k, "estados más similares a D.C. son:", paste(nombres_vecinos, collapse=", ")))

#Step 3: Impute with the mean of the 5 nearest neighbours

# 6. Calculate the mean of the neighbours for each year
imputacion_por_anio <- ev_adoption_14 %>%
  filter(state %in% nombres_vecinos) %>% # We filter only the data from the neighbouring states
  group_by(year) %>%                     # We group by year
  summarise(
    trucks_imputado = mean(Trucks, na.rm = TRUE) # We calculate the mean of 'Trucks' for each year
  )

print(imputacion_por_anio)

#Step 3: Perform the imputation with the vector calculated in the previous step
# 7. Join these imputation values to your original dataset and fill the gaps
ev_adoption_15 <- ev_adoption_14 %>%
  # We join the values to impute based on the year
  left_join(imputacion_por_anio, by = "year") %>%
  # We use mutate with ifelse to fill only the NAs for D.C.
  mutate(
    Trucks = ifelse(
      state == "District Of Columbia" & is.na(Trucks), # Condition
      trucks_imputado,                                  # Value if TRUE
      Trucks                                            # Value if FALSE
    )
  ) %>%
  select(-trucks_imputado) # We clean up the helper column

# 8. Verify the result for D.C.
ev_adoption_15 %>%
  filter(state == "District Of Columbia") %>%
  select(state, year, Trucks)

#Check normality and plot
ggplot(data = ev_adoption_15, aes(x=state, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="state", y="Number of trucks")
ggplot(data = ev_adoption_15, aes(x=year, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Year", y="Number of trucks")
ggplot(data = ev_adoption_15, aes(x=Index, y = Trucks))+ geom_line(aes(color = Trucks))+labs(x="Index", y="Number of trucks")
ggplot(data = ev_adoption_15, aes(x=Trucks))+ 
  geom_histogram(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Number of trucks", x="Trucks", y="Frequency") +
  theme_minimal()
ev_adoption_15 %>% plot_normality(Trucks) #It is a quasi-normal function
resultado_shapiro <- shapiro.test(ev_adoption_15$Trucks)
print(resultado_shapiro)

#variable Party
#I already have the vector of neighbours most similar to DC. I will use this information to impute the Party column by mode of the neighbours
#As it is a categorical variable, I only draw the bar chart
ggplot(data = ev_adoption_15, aes(x=Party))+ 
  geom_bar(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Political party in power", x="Party", y="Frequency") +
  theme_minimal()

# Calculate the mode of the neighbours for each year
ev_adoption_15 <- ev_adoption_15 %>%
  mutate(Party = na_if(Party,""))

imputacion_por_moda <- ev_adoption_16 %>%
  filter(state %in% nombres_vecinos) %>% # We filter only the data from the neighbouring states
  group_by(year) %>%                     # We group by year
  summarise(
    party_imputado = getmode(Party) # We calculate the mode of 'Party' for each year
  )

print(imputacion_por_moda)

# Join these imputation values to your original dataset and fill the gaps

ev_adoption_16 <- ev_adoption_15 %>%
  # We join the values to impute based on the year
  left_join(imputacion_por_moda, by = "year") %>%
  # We use mutate with ifelse to fill only the NAs for D.C.
  mutate(
    Party = ifelse(
      state == "District Of Columbia" & is.na(Party), # Condition
      party_imputado,                                  # Value if TRUE
      Party                                            # Value if FALSE
    )
  ) %>%
  select(-party_imputado) # We clean up the helper column

# Verify the result for D.C.
ev_adoption_16 %>%
  filter(state == "District Of Columbia") %>%
  select(state, year, Party)

#I have already done the 6 imputations for the District of Columbia that were missing for the years 2018 to 2023.
#Now we have to impute for all states for the years 2016 and 2017. I will impute by mode.

#I will try to impute by mode
ev_adoption_16 <- ev_adoption_16 %>%
  group_by(state) %>%
  mutate(
    Party= ifelse(
      is.na(Party),
      getmode(Party),
      Party
    )
  ) %>%
  ungroup()

#As it is a categorical variable, I only draw the bar chart
ggplot(data = ev_adoption_16, aes(x=Party))+ 
  geom_bar(bins=20, fill = "darkcyan", color= "white",alpha=0.9) +
  labs(title= "Political party in power", x="Party", y="Frequency") +
  theme_minimal() 

#Final cleaning of the table
#remove the Trucks and Trucks_Share columns for redundancy or inconsistency
borrar <- c("Total","Trucks_Share")
ev_adoption_17 <- ev_adoption_16[,!(names(ev_adoption_16) %in% borrar)]

#remove all columns created by the hot-deck imputation

ev_adoption_17 <- ev_adoption_17 %>% select(Index:Party)

write.csv(ev_adoption_17, file = ".../ev_adoption_clean.csv",row.names=FALSE)