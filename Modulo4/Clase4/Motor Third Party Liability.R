###MotorThirdPartyLiability(MTPL)

library(insurancerating)
library(dplyr)
library(ggplot2)
library(statmod)
library(scales)
library(tweedie)
library(tidymodels)
library(glmmTMB)
library(DHARMa)

#install.packages(c("insurancerating","dplyr","ggplot2","statmod","scales","tweedie","tidymodels"))

?MTPL

###Motor Third Party Liability (MTPL) portfolio (30,000 policyholders)

###Descripción
###La base de datos contiene las características de 30,000 asegurados de una cartera de seguros de responsabilidad civil de automóviles (MTPL) 
###en Países Bajos. Incluye información sobre región, siniestros, exposición y prima.

#Format

#A data frame containing 30,000 rows and 7 variables:

#age_policyholder: Edad del asegurado (en años)

#power: Potencia del motor del vehículo (en kilovatios)

#zip: Indicador de región donde viven los asegurados (0–3)

#nclaims: Numero de reclamaciones

#amount: Severidad del siniestro (en euros)

#bm: Nivel de bonus-malus (0–22). Los niveles más altos indican un peor historial de siniestralidad

#exposure: Exposición, expresada en años. Por ejemplo, 
#si un vehículo está asegurado desde el 1 de julio, la exposición es igual a 0.5 para ese año

head(MTPL)

str(MTPL)

glimpse(MTPL)

attach(MTPL)

table(nclaims)

Dat<-data.frame(table(nclaims))

ST<-data.frame(Frecuencia=Dat$Freq,numero=gl(5,1,labels=Dat$nclaims))

p<-ggplot(ST, aes(x = numero, y = Frecuencia,fill=factor(numero))) +
  geom_bar(stat = "identity") +
geom_text(aes(label = sprintf("%.0f",Frecuencia)), 
            vjust = -.5)+  
ggtitle("Número de reclamaciones")+
theme(plot.title = element_text(color="blue", size=20, face="bold.italic"))+theme(legend.position = "none")
p

p1<-ggplot(ST, aes(x = numero, y = Frecuencia,fill=numero)) +
  geom_bar(stat = "identity") + 
  geom_text(aes(label = sprintf("%.2f%%", Frecuencia/sum(Frecuencia) * 100)), 
            vjust = -.5)+
ggtitle("Porcentaje de reclamaciones")+
theme(plot.title = element_text(color="blue", size=20, face="bold.italic"))+theme(legend.position = "none")

p1

MTPL %>%
  group_by(nclaims) %>%
  summarise (n = n()) %>%
  mutate(prop = n / sum(n)) %>%
ggplot(aes(MTPL,x = nclaims, y = n)) +
    geom_col(fill = c("#CC0033", "#e319dc", "#cc4e00", "#cc0030", "#3600cc")) +
    geom_text(aes(label = paste0(n, " | ", signif(n / nrow(MTPL) * 100, digits = 3), '%')), nudge_y = 350) + ggtitle("Numero de reclamaciones: Frecuencias absoluta y relativa")
    theme_gray()

###Histogramas

MTPL |> pivot_longer((!zip), 
 names_to = "Variable", values_to="Score") |>
   ggplot(aes(x=Score)) + geom_histogram(aes(y = ..density..),bins=25,colour = 6, fill = "darkmagenta") +
     facet_wrap("Variable",ncol = 3,scales = "free" ) + theme_minimal()


MTPL |> pivot_longer((zip), 
        names_to = "Variable", values_to = "Score") |> 
   ggplot(aes(x=Score)) + geom_bar(colour = 1, fill = "darkmagenta") +
     facet_wrap("Variable",ncol = 1,scales = "free" ) + theme_minimal()

###box-plot

MTPL |> pivot_longer((!zip), 
  values_to="Score",names_to = "Variable") |>
    ggplot(aes(y=Score)) + geom_boxplot(aes(fill="darkred"),colour = 6, show.legend = FALSE) +
      facet_wrap("Variable",ncol = 3,scales = "free" ) + theme_minimal()

###Densidad

MTPL |> pivot_longer((!zip), 
   names_to = "Variable",values_to="Score") |>
     ggplot(aes(x=Score)) + geom_density(aes(fill="darkviolet"),colour = 6,show.legend = FALSE) +
       facet_wrap("Variable",ncol = 3,scales = "free" ) + theme_minimal()


###Analizando zip (region) como factor de riesgo
###factor_analysis: Realiza un análisis para factores de riesgo discretos en una cartera de seguros
###Se calculan los siguientes estadísticos descriptivos

#frequency = number of claims / exposure

#average severity = severity / number of claims

#risk premium = severity / exposure

#loss ratio = severity / premium

#average premium = premium / exposure

fa <- factor_analysis(
  MTPL,
  risk_factors = "zip",
  claim_count = "nclaims",
  exposure = "exposure",
  claim_amount = "amount"
)

fa

###"A mano"

a2<-MTPL|>filter(zip=="2")

sum(a2$nclaims)/sum(a2$exposure)

sum(a2$amount)/sum(a2$nclaims)

sum(a2$amount)/sum(a2$exposure)

###

autoplot(fa, metrics = c("exposure", "frequency", "risk_premium"))

###

#Continuous variables: Por qué se tratan por separado las variables continuas

#Por lo general, las variables continuas no se utilizan directamente para calcular una tarifa. En la práctica de tarificación, habitualmente:

#Se analizan como variables continuas

#Se convierten en segmentos tarifarios (esencialmente se segmentan o discretizan)

#Se utilizan en un GLM como factores de tarificación categóricos

#Esto garantiza que la tarifa final siga siendo interpretable e implementable. Este paso se utiliza para examinar:

#Patrones no lineales

#Volatilidad local

#Areas o regiones con baja exposición

#Vislumbrar puntos de corte plausibles para los segmentos tarifarios

###Aqui tenemos una variable "continua": age_policyholder
###Podemos ver cuál es su relacion funcional con el numero de reclamaciones

age_freq <- risk_factor_gam(
  data = MTPL,
  risk_factor = "age_policyholder",
  claim_count = "nclaims",
  exposure = "exposure"
)

autoplot(age_freq, show_observations = TRUE) 

###Y su relacion con la severidad

age_sev <- risk_factor_gam(
  data = MTPL,
  risk_factor = "age_policyholder",
  claim_count = "amount",
  exposure = "exposure"
)

autoplot(age_sev, show_observations = TRUE) 

#Determinación de segmentos tarifarios

age_segments <- derive_tariff_segments(age_freq)
autoplot(age_segments)

###Este procedimiento convierte la variable continua en segmentos tarifarios homogéneos en cuanto al riesgo

###Los segmentos resultantes deben reflejar diferencias de riesgo y, al mismo tiempo, 
###ser adecuados para su uso en el calculo de una tarifa

###Incorporación de segmentos tarifarios en los datos

dat <- MTPL |>
  add_tariff_segments(age_segments, name = "age_cat") |>
  mutate(across(where(is.character), as.factor)) |>
  mutate(across(where(is.factor), ~ set_reference_level(., exposure)))

head(dat)

###set_reference_level() establece el nivel de referencia en aquel con la mayor exposicion 
###En los modelos de tarificacion, esta suele ser la linea base mas estable e interpretable

###Ajuste o estimacion de un modelo GLM

###Por qué se utilizan los GLM en seguros

###Los modelos lineales generalizados (GLM) se utilizan ampliamente en la tarificación de seguros porque:

###Admiten distribuciones para respuesta no normales

###Producen efectos multiplicativos interpretables

###Pueden traducirse a tarifas

###Una modelacion habitual es:

#frecuencia –> GLM  Poisson
#severidad –> GLM Gamma

###Modelo de frecuencia (solo age_cat como predictora)

mod_freq <- glm(
  nclaims ~ age_cat,
  offset = log(exposure),
  family = poisson(),
  data = dat
)

mod_freq

summary(mod_freq)

###Bondad de ajuste del modelo de frecuencia

set.seed(123)
res_freq <- check_residuals(mod_freq, n_simulations = 250)

autoplot(res_freq)

check_overdispersion(mod_freq)

simulation_res <- simulateResiduals(fittedModel = mod_freq, plot = F)
plot(simulation_res)

testZeroInflation(mod_freq)

model_performance(mod_freq)

#Estabilidad predictiva
#Las medidas de rendimiento individuales proporcionan únicamente una estimación puntual. 
#En muchos contextos de pricing, también resulta relevante evaluar la estabilidad de dicho 
#rendimiento ante pequeñas variaciones en los datos

bootstrap_performance(mod_freq, dat, n_resamples = 100, show_progress = FALSE) |>
autoplot()


###Modelo de severidad (solo age_cat como predictora)

mod_sev <- glm(
  amount ~ age_cat,
  weights = nclaims,
  family = Gamma(link = "log"),
  data = dat |> filter(amount > 0)
)

mod_sev

summary(mod_sev)

###Bondad de ajuste del modelo de frecuencia

set.seed(123)
res_sev <- check_residuals(mod_sev, n_simulations = 250)
autoplot(res_sev)

simulation_res2 <- simulateResiduals(fittedModel = mod_sev, plot = F)
plot(simulation_res2)

model_performance(mod_sev)



###La frecuencia y la severidad se modelan por separado porque reflejan diferentes aspectos del proceso de pérdida

###Construcción de un modelo de prima

premium_df <- dat |>
  add_prediction(mod_freq, mod_sev) |>
  mutate(premium = pred_nclaims_mod_freq * pred_amount_mod_sev)

head(premium_df)

###Se estima la prima pura (premium), es decir, la pérdida esperada por unidad de exposicion

###Modelo de prima

prima_model <- glm(
  premium ~ age_cat + zip,
  weights = exposure,
  family = Gamma(link = "log"),
  data = premium_df
)

prima_model

summary(prima_model)

###Este modelo combina los factores de calificación en una única estructura de prima
###En la práctica, este suele ser el modelo que más se aproxima a la lógica tarifaria final, 
###ya que refleja el nivel de prima en lugar de solo componentes individuales del modelo, como la frecuencia o la severidad

rt <- rating_table(prima_model)
rt

#La función rating_table() expresa los coeficientes ajustados en función de los niveles originales de los factores, 
#incluido el nivel de referencia

#Este resultado se utiliza habitualmente para analizar las relaciones tarifarias

rating_table(prima_model) |>
  autoplot()

summary(prima_model)

model_performance(prima_model)

set.seed(123)
res_prima_model <- check_residuals(prima_model, n_simulations = 250)

autoplot(res_prima_model)

simulation_res_prima_model <- simulateResiduals(fittedModel = prima_model, plot = F)
plot(simulation_res_prima_model)

###Adicionando bonus malus (bm) al modelo final

prima_model2 <- glm(
  premium ~ age_cat + zip + bm,
  weights = exposure,
  family = Gamma(link = "log"),
  data = premium_df
)

prima_model2

summary(prima_model2)

#Observamos que la variable bonus malus (bm) no es significativa, así que NO TIENE SENTIDO contemplar un nuevo modelo con ella.

prima_model3 <- glm(
  premium ~ age_cat + zip + power,
  weights = exposure,
  family = Gamma(link = "log"),
  data = premium_df
)

prima_model3

summary(prima_model3)

####Resultados prima_model

library(car)
library(effects)

Anova(prima_model)

allEffects(prima_model)

plot(allEffects(prima_model))

prima_model






























