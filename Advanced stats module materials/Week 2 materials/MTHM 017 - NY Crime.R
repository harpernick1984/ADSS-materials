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


# Below is additional stretch material

# Recode this into distinct sections so that you can explore the diagnostics.

# ```{r}
#### NY Crime data

crime <- read.csv('nycrime.csv')
crime$THEFT[14] <- -14

## Plot data

ggplot(crime, aes(x = MAN, y = THEFT)) +
  geom_point(colour = '#aa0078', size = 2.5) +
  xlab('% change in manpower') +
  ylab('change in theft rate') +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14))

quantiles_95 <- function(x) {
  r <- quantile(x, probs=c(0.025, 0.25, 0.5, 0.75, 0.975))
  names(r) <- c("ymin", "lower", "middle", "upper", "ymax")
  r
}

ggplot(crime, aes(x = as.factor(DIST), y = THEFT)) +
  geom_boxplot(fill = '#0093af', alpha = 0.3) +
  # stat_summary(fun.data = quantiles_95, geom = "boxplot", fill = '#0093af', alpha = 0.4) +
  xlab('district') + ylab('change in theft rate') +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14))

set.seed(125)

# Define model
jags.mod14 <- function(){
  # likelihood
  for(i in 1:N14){
    THEFT14[i] ~ dnorm(mu14[i], tau14)
    mu14[i] <- alpha14 + beta14*MAN14[i] + delta14[DIST14[i]]
  }
  # priors
  alpha14 ~ dnorm(0, 1e-5)
  beta14 ~ dnorm(0, 1e-5)
  delta14[1] <- 0 # set coefficient for reference category to zero
  delta14[2] ~ dnorm(0, 1e-5)
  delta14[3] ~ dnorm(0, 1e-5)
  tau14 ~ dgamma(0.001, 0.001)
  sigma2_14 <- 1/tau14
}      

# the initial values list need to specify initial values for 
# delta14[2] and delta14[3], but not for delta14[1]. E.g.
inits14_1 <- list(alpha14 = 1, beta14 = -2, delta14 = c(NA, -2,4), tau14 = 2) 
inits14_2 <- list(alpha14 = -2, beta14 = 1, delta14 = c(NA, 1, -3), tau14 = 5) 

jags.inits14 <- list(inits14_1,inits14_2)

# parameters to monitor
jags.param14 <- c('alpha14','beta14','delta14')

# data
MAN14 <- crime$MAN
THEFT14 <- crime$THEFT
DIST14 <- crime$DIST
N14 <- length(MAN14)
jags.data14 <- list('MAN14','THEFT14','DIST14','N14')

# fit model
jags.mod.fit14 <- jags(data = jags.data14, inits = jags.inits14,
                     parameters.to.save = jags.param14, n.thin = 1,
                     n.chain = 2, n.burnin = 500, DIC = FALSE,
                     n.iter = 10000, model.file = jags.mod14)

# look at summary
print(jags.mod.fit14)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model48321062eb8.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 500 discarded)
# n.sims = 19000 iterations saved
#             mu.vect sd.vect   2.5%    25%    50%    75%  97.5%  Rhat n.eff
# alpha14      2.277   1.973  -1.592  0.996  2.293  3.558  6.199 1.001 19000
# beta14      -0.284   0.118  -0.517 -0.360 -0.284 -0.207 -0.050 1.001 19000
# delta14[1]   0.000   0.000   0.000  0.000  0.000  0.000  0.000 1.000     1
# delta14[2]   0.477   2.998  -5.462 -1.481  0.459  2.434  6.439 1.001 19000
# delta14[3]  -4.303   3.121 -10.475 -6.362 -4.282 -2.272  1.860 1.001 19000


# convert to mcmc object
jagsfit.mcmc14 <- as.mcmc(jags.mod.fit14)

par(mfrow = c(1, 1))
MCMCtrace(jagsfit.mcmc14,
          params = c('beta14'),
          type = 'density',
          ind = TRUE, 
          ISB = FALSE,
          pdf = FALSE,
          col_den = c("#aa0078", "#0093af"))

MCMCtrace(jagsfit.mcmc14,
          params = c('delta14\\[2\\]'),
          type = 'density',
          ind = TRUE, 
          ISB = FALSE,
          pdf = FALSE,
          col_den = c("#aa0078", "#0093af"))

MCMCtrace(jagsfit.mcmc14,
          params = c('delta14\\[3\\]'),
          type = 'density',
          ind = TRUE, 
          ISB = FALSE,
          pdf = FALSE,
          col_den = c("#aa0078", "#0093af"))

# plot density manually 

sim.values14 <- jags.mod.fit14$BUGSoutput$sims.list

df14 <- data.frame(beta14 = sim.values14$beta14,
                 delta14_2 = sim.values14$delta14[,2],
                 delta14_3 = sim.values14$delta14[,3])

p14_1 <- ggplot(data = df14, aes(x = beta14)) +
  geom_density(colour = '#0093af', fill = '#0093af', alpha = 0.3) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 18)) +
  geom_vline(xintercept = 0, linetype = "dashed", 
             colour = "#aa0078", size = 1) +
  ggtitle('Posterior density of beta14')

p14_2 <- ggplot(data = df14, aes(x = delta14_2)) +
  geom_density(colour = '#0093af', fill = '#0093af', alpha = 0.3) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 18)) +
  geom_vline(xintercept = 0, linetype = "dashed", 
             colour = "#aa0078", size = 1) +
  ggtitle('Posterior density of delta14[2]')

p14_3 <- ggplot(data = df14,aes(x = delta14_3)) +
  geom_density(colour = '#0093af', fill = '#0093af', alpha = 0.3) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 18)) +
  geom_vline(xintercept = 0, linetype = "dashed", 
             colour = "#aa0078", size = 1) +
  ggtitle('Posterior density of delta14[3]')

p14_1
p14_2
p14_3

########################################
# Try to remove DIST

jags.mod15 <- function(){
  # likelihood
  for(i in 1:N15){
    THEFT15[i] ~ dnorm(mu15[i], tau15)
    mu15[i] <- alpha15 + beta15*MAN15[i]
  }
  # priors
  alpha15 ~ dnorm(0, 1e-5)
  beta15 ~ dnorm(0, 1e-5)
  tau15 ~ dgamma(0.001, 0.001)
  sigma2_15 <- 1/tau15
} 

inits15_1 <- list(alpha15 = 1, beta15 = -2, tau15 = 2) 
inits15_2 <- list(alpha15 = -2, beta15 = 1, tau15 = 5) 

jags.inits15 <- list(inits15_1,inits15_2)

# parameters to monitor
jags.param15 <- c('alpha15','beta15','mu15')

# data
MAN15 <- crime$MAN
THEFT15 <- crime$THEFT
N15 <- length(MAN15)
jags.data15 <- list('MAN15','THEFT15','N15')

# fit model
jags.mod.fit15 <- jags(data = jags.data15, inits = jags.inits15,
                      parameters.to.save = jags.param15, n.thin = 1,
                      n.chain = 2, n.burnin = 500, DIC = FALSE,
                      n.iter = 10000, model.file = jags.mod15)


# look at summary
print(jags.mod.fit15)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model4831ce542f8.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 500 discarded)
# n.sims = 19000 iterations saved
# mu.vect sd.vect    2.5%     25%    50%    75%  97.5%  Rhat n.eff
# alpha15    1.636   1.362  -1.109   0.749  1.634  2.508  4.312 1.001 19000
# beta15    -0.219   0.109  -0.433  -0.289 -0.221 -0.149 -0.004 1.001 19000
# mu15[1]    5.095   1.696   1.732   3.980  5.108  6.203  8.447 1.001 19000
# mu15[2]    1.421   1.409  -1.413   0.509  1.417  2.321  4.204 1.001 19000
# mu15[3]    0.822   1.572  -2.315  -0.195  0.819  1.825  3.946 1.001 19000
# mu15[4]    2.815   1.242   0.335   2.019  2.814  3.621  5.263 1.001 19000
# mu15[5]    3.881   1.359   1.180   2.998  3.889  4.769  6.560 1.001 19000
# mu15[6]    3.462   1.289   0.911   2.628  3.474  4.294  6.000 1.001 19000
# mu15[7]    3.348   1.275   0.816   2.525  3.358  4.172  5.853 1.001 19000
# mu15[8]    0.150   1.795  -3.426  -1.017  0.154  1.300  3.730 1.001 19000
# mu15[9]    3.570   1.304   0.986   2.723  3.583  4.416  6.137 1.001 19000
# mu15[10]   3.734   1.331   1.088   2.870  3.749  4.603  6.358 1.001 19000
# mu15[11]   2.088   1.286  -0.508   1.263  2.089  2.919  4.614 1.001 19000
# mu15[12]   1.803   1.330  -0.878   0.941  1.801  2.657  4.423 1.001 19000
# mu15[13]   3.019   1.248   0.541   2.215  3.022  3.830  5.470 1.001 19000
# mu15[14]  -7.012   5.004 -16.905 -10.259 -7.047 -3.782  2.920 1.001 19000
# mu15[15]   4.004   1.385   1.258   3.105  4.013  4.906  6.725 1.001 19000
# mu15[16]   3.427   1.284   0.886   2.597  3.440  4.257  5.956 1.001 19000
# mu15[17]   2.255   1.267  -0.312   1.443  2.252  3.079  4.740 1.001 19000
# mu15[18]   5.189   1.728   1.769   4.053  5.204  6.321  8.611 1.001 19000
# mu15[19]   4.050   1.396   1.277   3.145  4.057  4.958  6.800 1.001 19000
# mu15[20]   4.840   1.613   1.653   3.787  4.852  5.897  8.013 1.001 19000
# mu15[21]   5.578   1.867   1.896   4.344  5.590  6.798  9.270 1.001 19000
# mu15[22]   1.469   1.398  -1.348   0.563  1.467  2.359  4.226 1.001 19000
# mu15[23]   4.000   1.384   1.253   3.100  4.010  4.901  6.717 1.001 19000


# convert to mcmc object
jagsfit.mcmc15 <- as.mcmc(jags.mod.fit15)

MCMCtrace(jagsfit.mcmc15,
          ind = TRUE, 
          pdf = FALSE)
# ```


