###TweedieMTPL2

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
p_hat <- p_profile$p.max
cat("Valor estimado de la potencia en Tweedie, p:", round(p_hat, 3), "\n\n")

fit_tw <- glm(amount ~ age_cat + zip, weights = exposure, family = statmod::tweedie(var.power=p_hat, link.power=0),data = dat)

summary(fit_tw)

###MUY POCOS COEFICIENTES SIGNIFICATIVOS ¿¿¿???

confint(fit_tw)

###Corremos el modelo tweedie con otra libreria

fit_tw2.1 = glmmTMB(amount ~ age_cat + zip, weights = exposure, family = tweedie(link = "log"), data=dat)
summary(fit_tw2.1)

confint(fit_tw2.1)

family_params(fit_tw2.1)

###Opcion para ver el estimador de p

fit_tw3 <- gam(amount ~ age_cat + zip, weights = exposure, data=dat, family = tw(link = "log"))
summary(fit_tw3)

###Trabajeremos en adelante con fit_tw2.1

###Bondad de ajuste

simres_tw2.1 = simulateResiduals(fit_tw2.1)

plot(simres_tw2.1)

model_performance(fit_tw2.1)

testDispersion(simres_tw2.1)

testOutliers(simres_tw2.1)

######################################

fit_tw4 = glmmTMB(amount ~ age_cat + zip + bm, weights = exposure, family = tweedie(link = "log"), data=dat)
summary(fit_tw4)

family_params(fit_tw4)

fit_tw5 = glmmTMB(amount ~ age_cat + zip + power, weights = exposure, family = tweedie(link = "log"), data=dat)
summary(fit_tw5)

family_params(fit_tw5)

fit_tw6 = glmmTMB(amount ~ age_cat + zip + power + bm, weights = exposure, family = tweedie(link = "log"), data=dat)
summary(fit_tw6)

family_params(fit_tw6)

###Parece ser que el mejor modelo involucra solo la edad discreta y la zona 

fit_tw2.1 = glmmTMB(amount ~ age_cat + zip , weights = exposure, family = tweedie(link = "log"), data=dat)
summary(fit_tw2.1)

###Bondad de ajuste: YA LA HICIMOS


###Modelo con potencia discretizada

power_freq <- risk_factor_gam(
  data = MTPL,
  risk_factor = "power",
  claim_count = "nclaims",
  exposure = "exposure"
)

autoplot(power_freq, show_observations = TRUE) ###No parece una relación distinta de la lineal

power_segments <- derive_tariff_segments(power_freq)
autoplot(power_segments)  ###Se confirma que la relación es lineal, y no tiene sentido segmentar

###

library(car)
library(effects)

Anova(fit_tw2.1)

allEffects(fit_tw2.1)

plot(allEffects(fit_tw2.1))

help("allEffects")
#####################################

dat$fitted<-fitted(fit_tw2.1)
head(dat)

agg<-aggregate(cbind(amount,fitted)~zip, data=dat, FUN=mean)
bp<-barplot(t(as.matrix(agg[,-1])),beside=TRUE,names.arg=agg$zip,
            col=c("#822453","#48A8A8"),ylim=c(0,max(agg[,-1])*1.2),
            ylab="Severidad media",main="Severidad observada vs. tweedie severidad predicha por zona")
            legend("topright",c("Observado","Ajustado"),fill=c("#822453","#48A8A8"),bty="n")

agg1<-aggregate(cbind(amount,fitted)~age_cat, data=dat, FUN=mean)
bp1<-barplot(t(as.matrix(agg1[,-1])),beside=TRUE,names.arg=agg1$age_cat,
            col=c("#822453","#48A8A8"),ylim=c(0,max(agg1[,-1])*1.2),
            ylab="Severidad media",main="Severidad observada vs. tweedie severidad predicha por categoría de edad")
            legend("topright",c("Observado","Ajustado"),fill=c("#822453","#48A8A8"),bty="n")


































