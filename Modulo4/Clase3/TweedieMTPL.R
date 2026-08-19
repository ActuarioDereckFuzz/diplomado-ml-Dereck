###TweedieMTPL

library(insurancerating)
library(dplyr)
library(ggplot2)
library(statmod)
library(scales)
library(tweedie)
library(tidymodels)
library(glmmTMB)
library(DHARMa)
library(mgcv)

###Esta modelacion Tweedie se realiza con la misma base del analisis en dos partes;
###misma finalidad y mismo preprocezamiento de la informacion

head(MTPL)

str(MTPL)

glimpse(MTPL)

attach(MTPL)

###La variable "continua" age_policyholder se podria incluir a traves de un modelo
###GAM (Generalized Additive Model), pero la vamos a discretizar igual que se realizo antes
###además de incorporarla a la base ya discretizada

age_freq <- risk_factor_gam(
  data = MTPL,
  risk_factor = "age_policyholder",
  claim_count = "nclaims",
  exposure = "exposure"
)

autoplot(age_freq, show_observations = TRUE) 

age_segments <- derive_tariff_segments(age_freq)
autoplot(age_segments)

dat <- MTPL |>
  add_tariff_segments(age_segments, name = "age_cat") |>
  mutate(across(where(is.character), as.factor)) |>
  mutate(across(where(is.factor), ~ set_reference_level(., exposure)))

head(dat)

dim(dat)

###Ajuste Tweedie

###Estimacion del parametro de potencia, p

library(tweedie)
library(statmod)

p_profile <- tweedie.profile(amount ~ 1,
                        data = dat, p.vec = seq(1.2, 1.8, by = 0.1),
                        do.plot = FALSE, method = "series")
plot(p_profile$y)
p_hat <- p_profile$p.max
cat("Valor estimado de la potencia en Tweedie, p:", round(p_hat, 3), "\n\n")

fit_tw <- glm(amount ~ age_cat + zip, weights = exposure, family = statmod::tweedie(var.power=p_hat, link.power=0),data = dat)

summary(fit_tw)

###MUY POCOS COEFICIENTES SIGNIFICATIVOS ¿¿¿???

confint(fit_tw)

###

tw_res = simulateResiduals(fit_tw) ###No funciona

###Corremos el modelo tweedie con otra libreria

fit_tw2 = glmmTMB(amount ~ age_cat + zip, offset=log(exposure), family = tweedie(link = "log"), data=dat)
summary(fit_tw2)

confint(fit_tw2)

family_params(fit_tw2)

fit_tw %>% tidy()

###Opcion para ver el estimador de p

fit_tw3 <- gam(amount ~ age_cat + zip, offset=log(exposure), data=dat, family = tw(link = "log"))
summary(fit_tw3)

###Trabajeremos en adelante con fit_tw2

###Bondad de ajuste

simres_tw2 = simulateResiduals(fit_tw2)

plot(simres_tw2)

model_performance(fit_tw2)

testDispersion(simres_tw2)

testOutliers(simres_tw2)



###############################

fit_tw4 = glmmTMB(amount ~ age_cat + zip + bm, offset=log(exposure), family = tweedie(link = "log"), data=dat)
summary(fit_tw4)

family_params(fit_tw4)
##############################

fit_tw5 = glmmTMB(amount ~ age_cat + zip + power, offset=log(exposure), family = tweedie(link = "log"), data=dat)
summary(fit_tw5)

family_params(fit_tw5)

model_performance(fit_tw5)

### Comparación con el modelo tw2 y tw5

anova(fit_tw5,fit_tw2)

#############################


fit_tw6 = glmmTMB(amount ~ age_cat + zip + power + bm, offset=log(exposure), family = tweedie(link = "log"), data=dat)
summary(fit_tw6)

family_params(fit_tw6)

model_performance(fit_tw6)


### Comparación con el modelo tw6 y tw5

anova(fit_tw5,fit_tw6) ## No es necesario agregar la variable bm

############################


###Bondad de ajuste

simres_tw5 = simulateResiduals(fit_tw5)

plot(simres_tw5)

model_performance(fit_tw5)

testDispersion(simres_tw5)

testOutliers(simres_tw5)












