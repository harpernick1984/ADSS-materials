# 1) Firstly load in a range of libraries
# ```{r}
library(ggplot2)
library(Rmisc)
library(tidyverse)
#library(TailRank)
library(R2jags)
library(coda)
library(lattice)
#library(jagsplot)
library(MCMCvis)
library(rjags)
#library(mcmcplots)
# ```

# Use the programming sequence as we did last week.
# a) Setting the jags.model. 
# b) Set the seed (uncomment to fix this). 
# c) Running the jags analysis. 
# d) Print the result output.
# ```{r}


# ```{r}
### Union density example

## Vague priors

# Model - centred

jags.mod16 <- function(){ 
  for(i in 1:20){
    Uden16[i] ~ dnorm(mu16[i], tau16) # normal likelihood
    # linear predictor with centered covariates
    mu16[i] <- b16_0 + b16[1] * (LeftG16[i] - mean(LeftG16[])) + 
      b16[2] * (LabF16[i] - mean(LabF16[])) + b16[3] * (IndC16[i] - mean(IndC16[]))
  }
  # vague prior on residual error variance
  tau16 ~ dgamma(0.001, 0.001)
  sigma16_2 <- 1/tau16 # residual error variance
  # vagues priors on regression coefficients
  b16_0 ~ dnorm(0, 0.00001) 
  for(k in 1:3){
    b16[k] ~ dnorm(0, 0.00001) 
  }
}

# Data 

data16 <- c(82.4,8.28,111.84,1.55,80.0,6.90,73.17,1.71, 
         74.2,4.39,17.25,2.06,73.3,7.62,59.33,1.56, 
         71.9,8.12,43.25,1.52,69.8,7.71,90.24,1.52, 
         68.1,6.79,0.00,1.75,65.6,7.81,48.67,1.53, 
         59.4,6.96,60.00,1.64,58.9,7.41,83.08,1.58, 
         51.4,8.60,33.74,1.37,50.6,9.67,0.00,0.86, 
         48.0,10.16,43.67,1.13,39.6,10.04,35.33,0.92, 
         37.7,8.41,31.50,1.25,35.4,7.81,11.87,1.68, 
         31.2,9.26,0.00,1.35,31.0,10.59,1.92,1.11, 
         28.2,9.84,8.67,0.95,24.5,11.44,0.00,1.00)
data16 <- matrix(data16, nrow = 20, ncol = 4, byrow = TRUE)
Uden16 <- data16[,1]
LabF16 <- data16[,2]
LeftG16 <- data16[,3]
IndC16 <- data16[,4]
jags.data16 <- list("Uden16","LabF16","LeftG16","IndC16")


# Parameters to monitor 
jags.param16 <- c("b16_0","b16")


# Specify initial values 
inits16_1 <- list("tau16" = 100, "b16_0" = 20, "b16" = c(10, -5, -25)) 
inits16_2 <- list("tau16" = 100000, "b16_0" = -100, "b16" = c(-100, 100, 500)) 
jags.inits16 <- list(inits16_1, inits16_2)

# fit the model
jags.mod.fit16 <- jags(data = jags.data16, inits = jags.inits16,
                     parameters.to.save = jags.param16, n.chains = 2, 
                     n.iter = 10000, n.burnin = 1000, n.thin = 1, 
                     model.file = jags.mod16)

print(jags.mod.fit16)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpHM4dXi/model1fa156afd95.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 1000 discarded)
# n.sims = 18000 iterations saved
#           mu.vect sd.vect  2.5%     25%     50%     75%   97.5%  Rhat n.eff
# b16[1]     0.273   0.235   0.111   0.218   0.271   0.324   0.434 1.249 17000
# b16[2]    -6.479   4.973 -14.551  -9.126  -6.462  -3.774   1.579 1.048  4700
# b16[3]     0.255  21.317 -40.865 -13.220   0.349  13.975  41.017 1.002  3500
# b16_0     54.047   3.595  49.139  52.446  54.044  55.635  58.992 1.115 18000
# deviance 151.718   4.482 146.863 148.964 150.938 153.546 160.878 1.010  3800



jags.mcmc16 <- as.mcmc(jags.mod.fit16)
summary(jags.mcmc16)
# Iterations = 1001:10000
# Thinning interval = 1 
# Number of chains = 2 
# Sample size per chain = 9000 

# 1. Empirical mean and standard deviation for each variable, plus standard error of the mean:
#             Mean      SD    Naive SE    Time-series SE
# b16_0     54.0466  3.5947   0.026794       0.018729
# b16[1]     0.2730  0.2351   0.001753       0.001586
# b16[2]    -6.4792  4.9731   0.037067       0.032875
# b16[3]     0.2552 21.3167   0.158886       0.155151
# deviance 151.7184  4.4821   0.033408       0.051751

# 2. Quantiles for each variable:
#           2.5%      25%      50%      75%    97.5%
# b16_0     49.139  52.4458  54.0440  55.6354  58.9916
# b16[1]     0.111   0.2185   0.2712   0.3238   0.4341
# b16[2]   -14.551  -9.1261  -6.4623  -3.7742   1.5791
# b16[3]   -40.865 -13.2200   0.3490  13.9745  41.0173
# deviance 146.863 148.9640 150.9382 153.5460 160.8782


sim.values16 <- jags.mod.fit16$BUGSoutput$sims.list$b16
df16 <- data.frame(LeftG16 = sim.values16[,1], LabF16 = sim.values16[,2], 
                   IndC16 = sim.values16[,3]) %>%
  gather(key = 'variable', value = 'b16')

df16$variable <- factor(df16$variable, levels = c('LeftG16', 'LabF16', 'IndC16'))

quantiles16_95 <- function(x) {
  r16 <- quantile(x, probs = c(0.025, 0.25, 0.5, 0.75, 0.975))
  names(r16) <- c("ymin", "lower", "middle", "upper", "ymax")
  r16
}

ggplot(df16, aes(x = variable, y = b16)) + 
  #  geom_boxplot(outlier.shape = NA,fill='#0093af',alpha=0.3) +
  #  scale_x_discrete(limits=unique(df$variable)) +
  stat_summary(fun.data = quantiles16_95, geom = "boxplot", fill = '#0093af', alpha = 0.4) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16),
        plot.title = element_text(size = 18)) +
  ylim(c(-45, 45)) + scale_y_continuous(breaks = c(-40, -20, 0, 20, 40)) +
  ggtitle('Posterior distribution of regression coefficients')



# ```

# ```{r}
### Wallerstein informative prior

# Model

jags.mod.w17 <- function(){ 
  for(i in 1:20){
    Uden17[i] ~ dnorm(mu17[i], tau17) # normal likelihood
    # linear predictor with centered covariates
    mu17[i] <- b17_0 + b17[1] * (LeftG17[i] - mean(LeftG17[])) + 
      b17[2] * (LabF17[i] - mean(LabF17[])) + b17[3] * (IndC17[i] - mean(IndC17[]))
  }
  # vague prior on residual error variance
  tau17 ~ dgamma(0.001, 0.001)
  sigma17_2 <- 1/tau17 # residual error variance
  # vagues priors on regression coefficients
  b17_0 ~ dnorm(0, 0.00001) 
  
  b17[1] ~ dnorm(0.3, 1/(0.15^2)) 
  b17[2] ~ dnorm(-5, 1/(2.5^2)) 
  b17[3] ~ dnorm(0, 0.00001) 
}


# Data 
data17 <- c(82.4,8.28,111.84,1.55,80.0,6.90,73.17,1.71, 
            74.2,4.39,17.25,2.06,73.3,7.62,59.33,1.56, 
            71.9,8.12,43.25,1.52,69.8,7.71,90.24,1.52, 
            68.1,6.79,0.00,1.75,65.6,7.81,48.67,1.53, 
            59.4,6.96,60.00,1.64,58.9,7.41,83.08,1.58, 
            51.4,8.60,33.74,1.37,50.6,9.67,0.00,0.86, 
            48.0,10.16,43.67,1.13,39.6,10.04,35.33,0.92, 
            37.7,8.41,31.50,1.25,35.4,7.81,11.87,1.68, 
            31.2,9.26,0.00,1.35,31.0,10.59,1.92,1.11, 
            28.2,9.84,8.67,0.95,24.5,11.44,0.00,1.00)
data17 <- matrix(data17, nrow = 20, ncol = 4, byrow = TRUE)
Uden17 <- data17[,1]
LabF17 <- data17[,2]
LeftG17 <- data17[,3]
IndC17 <- data17[,4]
jags.data17 <- list("Uden17","LabF17","LeftG17","IndC17")

# Parameters to monitor 
jags.param17 <- c("b17_0","b17")


# Specify initial values 
inits17_1 <- list("tau17" = 100, "b17_0" = 20, "b17" = c(10, -5, -25)) 
inits17_2 <- list("tau17" = 100000, "b17_0" = -100, "b17" = c(-100, 100, 500)) 
jags.inits17 <- list(inits17_1, inits17_2)


jags.mod.fit.w17 <- jags(data = jags.data17, inits = jags.inits17,
                       parameters.to.save = jags.param17, n.chains = 2, 
                       n.iter = 10000, n.burnin = 1000, n.thin = 1, 
                       model.file = jags.mod.w17)

print(jags.mod.fit.w17)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpHM4dXi/model1fa348bb3b4.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 1000 discarded)
# n.sims = 18000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# b17[1]     0.277   0.069   0.141   0.232   0.277   0.322   0.414 1.001  4600
# b17[2]    -5.406   2.081  -9.430  -6.823  -5.407  -4.023  -1.275 1.001 18000
# b17[3]     4.836  13.151 -19.611  -3.683   4.825  13.338  30.159 1.002 18000
# b17_0     54.021   3.388  49.242  52.471  54.051  55.623  58.827 1.001 15000
# deviance 150.586   3.741 146.623 148.337 149.928 152.073 158.283 1.001 18000


sim.values.w17 <- jags.mod.fit.w17$BUGSoutput$sims.list$b17
df.w17 <- data.frame(LeftG17 = sim.values.w17[,1], 
                     LabF17 = sim.values.w17[,2], 
                     IndC17 = sim.values.w17[,3]) %>%
  gather(key = 'variable', value = 'b17')


ggplot(df.w17, aes(x = variable, y = b17)) + 
  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.3) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16)) +
  ylim(c(-35, 35)) 


### Stephens informative prior

# Model

jags.mod.s18 <- function(){ 
  for(i in 1:20){
    Uden18[i] ~ dnorm(mu18[i], tau18) # normal likelihood
    # linear predictor with centered covariates
    mu18[i] <- b18_0 + b18[1] * (LeftG18[i] - mean(LeftG18[])) + 
      b18[2] * (LabF18[i] - mean(LabF18[])) + b18[3] * (IndC18[i] - mean(IndC18[]))
  }
  # prior on residual error variance
  tau18 ~ dgamma(0.001, 0.001)
  sigma18_2 <- 1/tau18 # residual error variance
  # vagues priors on regression coefficients
  b18_0 ~ dnorm(0, 0.00001) 
  
  b18[1] ~ dnorm(0.3, 1/(0.15^2)) 
  b18[2] ~ dnorm(0, 0.00001) 
  b18[3] ~ dnorm(10, 1/(5^2)) 
}

# Data 
data18 <- c(82.4,8.28,111.84,1.55,80.0,6.90,73.17,1.71, 
            74.2,4.39,17.25,2.06,73.3,7.62,59.33,1.56, 
            71.9,8.12,43.25,1.52,69.8,7.71,90.24,1.52, 
            68.1,6.79,0.00,1.75,65.6,7.81,48.67,1.53, 
            59.4,6.96,60.00,1.64,58.9,7.41,83.08,1.58, 
            51.4,8.60,33.74,1.37,50.6,9.67,0.00,0.86, 
            48.0,10.16,43.67,1.13,39.6,10.04,35.33,0.92, 
            37.7,8.41,31.50,1.25,35.4,7.81,11.87,1.68, 
            31.2,9.26,0.00,1.35,31.0,10.59,1.92,1.11, 
            28.2,9.84,8.67,0.95,24.5,11.44,0.00,1.00)
data18 <- matrix(data18, nrow = 20, ncol = 4, byrow = TRUE)
Uden18 <- data18[,1]
LabF18 <- data18[,2]
LeftG18 <- data18[,3]
IndC18 <- data18[,4]
jags.data18 <- list("Uden18","LabF18","LeftG18","IndC18")

# Parameters to monitor 
jags.param18 <- c("b18_0","b18")


# Specify initial values 
inits18_1 <- list("tau18" = 100, "b18_0" = 20, "b18" = c(10, -5, -25)) 
inits18_2 <- list("tau18" = 100000, "b18_0" = -100, "b18" = c(-100, 100, 500)) 
jags.inits18 <- list(inits18_1, inits18_2)


jags.mod.fit.s18 <- jags(data = jags.data18, inits = jags.inits18,
                       parameters.to.save = jags.param18, n.chains = 2, 
                       n.iter = 10000, n.burnin = 1000, n.thin = 1, 
                       model.file = jags.mod.s18)

print(jags.mod.fit.s18)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpHM4dXi/model1fa3fc3095f.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 1000 discarded)
# n.sims = 18000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# b18[1]     0.273   0.068   0.141   0.229   0.272   0.318   0.409 1.001  7900
# b18[2]    -4.793   4.007  -8.309  -5.962  -4.780  -3.566  -1.123 1.214 18000
# b18[3]     9.398   4.829   0.102   6.132   9.342  12.598  19.050 1.001 13000
# b18_0     54.027   5.852  49.384  52.541  54.081  55.602  58.759 1.226 18000
# deviance 150.435   3.776 146.636 148.217 149.745 151.888 157.810 1.008 18000


sim.values.s18 <- jags.mod.fit.s18$BUGSoutput$sims.list$b18
df.s18 <- data.frame(LeftG18 = sim.values.s18[,1],
                     LabF18 = sim.values.s18[,2],
                     IndC18 = sim.values.s18[,3]) %>%
  gather(key = 'variable', value = 'b18')


ggplot(df.s18, aes(x = variable, y = b18)) + 
  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.3) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16)) +
  ylim(c(-15, 25)) 




###### Boxplots

quantiles16_17_18_95 <- function(x) {
  r16_17_18 <- quantile(x, probs = c(0.025, 0.25, 0.5, 0.75, 0.975))
  names(r16_17_18) <- c("ymin", "lower", "middle", "upper", "ymax")
  r16_17_18
}

df.indc16_17_18 <- data.frame(Vague = sim.values16[,3],
                              Wallerstein = sim.values.w17[,3],
                              Stephens = sim.values.s18[,3]) %>%
  gather(key = 'variable', value = 'prior')

df.indc16_17_18$variable <- factor(df.indc16_17_18$variable, 
                                   levels = c('Vague','Wallerstein','Stephens'))

ggplot(df.indc16_17_18, aes(x = variable, y = prior)) + 
  #geom_boxplot(outlier.shape = NA, fill = '#0093af',alpha = 0.3) +
  stat_summary(fun.data = quantiles16_17_18_95, geom = "boxplot", fill = '#0093af', alpha = 0.4) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16),
        plot.title = element_text(size = 18))+
  ylim(c(-53, 53)) + ggtitle('Effect of IndC (Stephens hypothesis)') +
  scale_y_continuous(breaks = c(-40, -20, 0, 20, 40))


df.labf16_17_18 <- data.frame(Vague = sim.values16[,2],
                              Wallerstein = sim.values.w17[,2],
                              Stephens = sim.values.s18[,2]) %>%
  gather(key = 'variable', value = 'prior')

df.labf16_17_18$variable <- factor(df.labf16_17_18$variable,
                                   levels = c('Vague','Wallerstein','Stephens'))

ggplot(df.labf16_17_18, aes(x = variable, y = prior)) + 
  #  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.3) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16),
        plot.title = element_text(size = 18)) +
  ylim(c(-18,8)) + ggtitle('Effect of LabF (Wallerstein hypothesis)') +
  #  stat_boxplot(geom ='errorbar',width=0.6)
  stat_summary(fun.data = quantiles16_17_18_95, geom="boxplot", fill = '#0093af', alpha = 0.4) +
  scale_y_continuous(breaks = c(-15, -10, -5, 0, 5))


df.leftg16_17_18 <- data.frame(Vague = sim.values16[,1],
                               Wallerstein = sim.values.w17[,1],
                               Stephens = sim.values.s18[,1]) %>%
  gather(key = 'variable', value = 'prior')

df.leftg16_17_18$variable <- factor(df.leftg16_17_18$variable, 
                                    levels = c('Vague','Wallerstein','Stephens'))


ggplot(df.leftg16_17_18, aes(x = variable, y = prior)) + 
  #  stat_boxplot(geom = 'errorbar', width = 0.6) +
  #  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.9) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  stat_summary(fun.data = quantiles16_17_18_95, geom = "boxplot", fill = '#0093af', alpha = 0.4) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16)) +
  ylim(c(0, 0.5)) + ggtitle('LeftG')



df16_17_18 <- data.frame(Uden = Uden16, LabF = LabF16, IndC = IndC16, LeftG = LeftG16)

# ```


### Stephens informative prior with mean 0.1, sd <= 0.05

# Model

jags.mod.s19 <- function(){ 
  for(i in 1:20){
    Uden19[i] ~ dnorm(mu19[i], tau19) # normal likelihood
    # linear predictor with centered covariates
    mu19[i] <- b19_0 + b19[1] * (LeftG19[i] - mean(LeftG19[])) + 
      b19[2] * (LabF19[i] - mean(LabF19[])) + b19[3] * (IndC19[i] - mean(IndC19[]))
  }
  # prior on residual error variance
  tau19 ~ dgamma(0.001, 0.001)
  sigma19_2 <- 1/tau19 # residual error variance
  # vagues priors on regression coefficients
  b19_0 ~ dnorm(0, 0.00001) 
  
  b19[1] ~ dnorm(0.3, 1/(0.15^2)) 
  b19[2] ~ dnorm(0, 0.00001) 
  b19[3] ~ dnorm(0.1, 1/(0.05^2)) 
}

# Data 
data19 <- c(82.4,8.28,111.84,1.55,80.0,6.90,73.17,1.71, 
            74.2,4.39,17.25,2.06,73.3,7.62,59.33,1.56, 
            71.9,8.12,43.25,1.52,69.8,7.71,90.24,1.52, 
            68.1,6.79,0.00,1.75,65.6,7.81,48.67,1.53, 
            59.4,6.96,60.00,1.64,58.9,7.41,83.08,1.58, 
            51.4,8.60,33.74,1.37,50.6,9.67,0.00,0.86, 
            48.0,10.16,43.67,1.13,39.6,10.04,35.33,0.92, 
            37.7,8.41,31.50,1.25,35.4,7.81,11.87,1.68, 
            31.2,9.26,0.00,1.35,31.0,10.59,1.92,1.11, 
            28.2,9.84,8.67,0.95,24.5,11.44,0.00,1.00)
data19 <- matrix(data19, nrow = 20, ncol = 4, byrow = TRUE)
Uden19 <- data19[,1]
LabF19 <- data19[,2]
LeftG19 <- data19[,3]
IndC19 <- data19[,4]
jags.data19 <- list("Uden19","LabF19","LeftG19","IndC19")

# Parameters to monitor 
jags.param19 <- c("b19_0","b19")


# Specify initial values 
inits19_1 <- list("tau19" = 100, "b19_0" = 20, "b19" = c(10, -5, -25)) 
inits19_2 <- list("tau19" = 100000, "b19_0" = -100, "b19" = c(-100, 100, 500)) 
jags.inits19 <- list(inits19_1, inits19_2)


jags.mod.fit.s19 <- jags(data = jags.data19, inits = jags.inits19,
                         parameters.to.save = jags.param19, n.chains = 2, 
                         n.iter = 10000, n.burnin = 1000, n.thin = 1, 
                         model.file = jags.mod.s19)

print(jags.mod.fit.s19)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpHM4dXi/model1fa1d9c5cad.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 1000 discarded)
# n.sims = 18000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# b19[1]     0.278   0.068   0.145   0.234   0.277   0.321   0.412 1.001 18000
# b19[2]    -6.449   3.401  -9.501  -7.458  -6.450  -5.422  -3.265 1.208 18000
# b19[3]     0.100   0.051   0.000   0.066   0.100   0.134   0.199 1.001 18000
# b19_0     54.023   5.285  49.368  52.508  54.063  55.581  58.770 1.211 17000
# deviance 150.095   3.801 146.374 147.870 149.379 151.519 157.566 1.009 18000


sim.values.s19 <- jags.mod.fit.s19$BUGSoutput$sims.list$b19
df.s19 <- data.frame(LeftG19 = sim.values.s19[,1],
                     LabF19 = sim.values.s19[,2],
                     IndC19 = sim.values.s19[,3]) %>%
  gather(key = 'variable', value = 'b19')


ggplot(df.s19, aes(x = variable, y = b19)) + 
  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.3) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16)) +
  ylim(c(-15, 25)) 




###### Boxplots

quantiles16_17_19_95 <- function(x) {
  r16_17_19 <- quantile(x, probs = c(0.025, 0.25, 0.5, 0.75, 0.975))
  names(r16_17_19) <- c("ymin", "lower", "middle", "upper", "ymax")
  r16_17_19
}

df.indc16_17_19 <- data.frame(Vague = sim.values16[,3],
                              Wallerstein = sim.values.w17[,3],
                              Stephens = sim.values.s19[,3]) %>%
  gather(key = 'variable', value = 'prior')

df.indc16_17_19$variable <- factor(df.indc16_17_19$variable, 
                                   levels = c('Vague','Wallerstein','Stephens'))

ggplot(df.indc16_17_19, aes(x = variable, y = prior)) + 
  #geom_boxplot(outlier.shape = NA, fill = '#0093af',alpha = 0.3) +
  stat_summary(fun.data = quantiles16_17_19_95, geom = "boxplot", fill = '#0093af', alpha = 0.4) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16),
        plot.title = element_text(size = 18))+
  ylim(c(-53, 53)) + ggtitle('Effect of IndC (Stephens hypothesis)') +
  scale_y_continuous(breaks = c(-40, -20, 0, 20, 40))


df.labf16_17_19 <- data.frame(Vague = sim.values16[,2],
                              Wallerstein = sim.values.w17[,2],
                              Stephens = sim.values.s19[,2]) %>%
  gather(key = 'variable', value = 'prior')

df.labf16_17_19$variable <- factor(df.labf16_17_19$variable,
                                   levels = c('Vague','Wallerstein','Stephens'))

ggplot(df.labf16_17_19, aes(x = variable, y = prior)) + 
  #  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.3) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16),
        plot.title = element_text(size = 18)) +
  ylim(c(-18,8)) + ggtitle('Effect of LabF (Wallerstein hypothesis)') +
  #  stat_boxplot(geom ='errorbar',width=0.6)
  stat_summary(fun.data = quantiles16_17_19_95, geom="boxplot", fill = '#0093af', alpha = 0.4) +
  scale_y_continuous(breaks = c(-15, -10, -5, 0, 5))


df.leftg16_17_19 <- data.frame(Vague = sim.values16[,1],
                               Wallerstein = sim.values.w17[,1],
                               Stephens = sim.values.s19[,1]) %>%
  gather(key = 'variable', value = 'prior')

df.leftg16_17_19$variable <- factor(df.leftg16_17_19$variable, 
                                    levels = c('Vague','Wallerstein','Stephens'))


ggplot(df.leftg16_17_19, aes(x = variable, y = prior)) + 
  #  stat_boxplot(geom = 'errorbar', width = 0.6) +
  #  geom_boxplot(outlier.shape = NA, fill = '#0093af', alpha = 0.9) +
  #  scale_x_discrete(limits = unique(df$variable)) +
  stat_summary(fun.data = quantiles16_17_19_95, geom = "boxplot", fill = '#0093af', alpha = 0.4) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 16)) +
  ylim(c(0, 0.5)) + ggtitle('LeftG')



df16_17_19 <- data.frame(Uden = Uden16, LabF = LabF16, IndC = IndC16, LeftG = LeftG16)

# ```

