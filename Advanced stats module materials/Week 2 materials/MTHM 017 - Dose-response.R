# ---
# title: "MTHM017 Advanced Methods Workshop  B"
# author: "John T Bruun"
# date: "27/01/2021"
# output:
# pdf_document: default
# html_document: default
# word_document: default
# ---
  
# ```{r setup, include=FALSE}
# knitr::opts_chunk$set(echo = TRUE)
# ```
# Below is the script coding. 

# Extend your coding experience from last week. Use and adapt this code by altering the simulation length 
# as is and to reproduce the lecture note results. Remember the help()facility, the additional settings you 
# can alter in jags(), especially the number of chains, number of iterations and burn length.

# Task A: explore the dose – response Lecture 3 example. 
# The key aspect of using MCMC methods is to establish chain convergence. 
# Make yourself familiar with the jags.mod, jags.dat and initialisation so you can run these. Firstly use the set.seed functionality to obtain the same results (compare these in your groups). Then use a random seed. 

# Look at the Gelman-Rubin diagnostic (with 3 chains). How long do these chains need to be for the scale reduction factor to be < 1.1? 
  
# Task B: non-linear regression Dugon Lecture 4 example
# Examine the setting up of this problem. Alter the code to obtain predictions at age 25 and 33.  
# How do these look compared to the  predictions at age 35 and 40? How good are the convergence 
# diagnostics? Use a fixed and random seed also and compare your results in your groups. 

# Additional stretch: additional lecture examples are given. Practice coding this with comments so you can explain the steps clearly to one of your peers. 


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

# a) Lecture notes code
### dose-response model

set.seed(9731)

# Data
# dose
x4 <- c(1.69,1.72,1.75,1.78,1.81,1.83,1.86,1.88)
# number of subjects
n4 <- c(59,60,62,56,63,59,62,60)
# number of responders
r4 <- c(6,13,18,28,52,53,61,60)

N4 <- length(x4)

# ```


# ```{r}
# Define model
jags.mod4 <- function(){
  # likelihood
  for(i in 1:N4){
    r4[i] ~ dbin(p4[i],n4[i])
    logit(p4[i]) <- alpha4 + beta4*(x4[i]-mean(x4))
  }
  # priors
  alpha4 ~ dnorm(0,1/10000)
  beta4 ~ dnorm(0,1/10000)
}      
# ```
# ```{r}
# b) set.seed(2) # done above - but this is also a good place to keep this for repeated burn and chain length experimentation
# set.seed(Sys.time())
# ```

# ```{r}
# initial values for 2 chains
inits4_1 <- list(alpha4=-5, beta4=10) 
inits4_2 <- list(alpha4=5, beta4=-10)

jags.inits4 <- list(inits4_1,inits4_2)
# ```

# ```{r}
# parameters to monitor
jags.param4 <- c("alpha4","beta4")
# ```

# ```{r}
# data
jags.data4 <- list('x4','r4','n4','N4')
# ```


# ```{r}
# c) run the analysis (here is where you change the burn and iterations)

jags.mod.fit4 <- jags(data = jags.data4, inits = jags.inits4,
                     parameters.to.save = jags.param4, n.thin = 1,
                     n.chain = 2, n.burnin = 0, DIC = FALSE,
                     n.iter = 20000, model.file = jags.mod4)

#,n.burnin=50)
# ```
# ```{r}
# d) Print the jags results
jags.mod.fit4

# look at summary
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model4836cb49e7.txt", fit using jags,
# 2 chains, each with 20000 iterations (first 0 discarded)
# n.sims = 40000 iterations saved
#         mu.vect sd.vect  2.5%    25%   50%    75%  97.5%  Rhat n.eff
# alpha4   0.763   0.153  0.495  0.668  0.76  0.854  1.038 1.004   600
# beta4   34.645   2.935 29.003 32.703 34.62 36.560 40.488 1.001  4100


# ```
# Diagnostics: Look at the chain outputs. Firstly extract the chain record and also map this into a time series object. Here histogram, chain time series record together with auto and partial correlation are explored. What do the acf terms help us to assess?
# ```{r}

alpha.diag4 <- jags.mod.fit4$BUGSoutput$sims.list$alpha4
alpha.diag.ts4 <- ts(alpha.diag4)

histogram(alpha.diag4, xlab = "alpha4", plot = TRUE)
ts.plot(alpha.diag.ts4, xlab = "Iteration", ylab = "alpha4")
acf(alpha.diag.ts4)
acf(alpha.diag.ts4, type = "partial")

# ```

# Diagnostics: Now, as for before, look at the chain output for beta
# ```{r}

beta.diag4 <- jags.mod.fit4$BUGSoutput$sims.list$beta4
beta.diag.ts4 <- ts(beta.diag4)

histogram(beta.diag4, xlab = "beta4", plot = TRUE)
ts.plot(beta.diag.ts4, xlab = "Iteration", ylab = "beta4")
acf(beta.diag.ts4)
acf(beta.diag.ts4, type = "partial")

# ```

# ```{r}
# convert to mcmc object
jagsfit.mcmc4 <- as.mcmc(jags.mod.fit4)

# bug in R2jags --> keeps burnin even when burn-in has a high value
traceplot(jagsfit.mcmc4, parms = c("alpha4", "beta4"))

# this works with the bug as well
MCMCtrace(jagsfit.mcmc4,
          type = 'trace',
          ind = TRUE, 
          pdf = FALSE)

# ```

# ```{r}
# We can produce traceplot and density plot manually
sim.values4 <- jags.mod.fit4$BUGSoutput$sims.array

alpha4_1 <- sim.values4[,1,1]
alpha4_2 <- sim.values4[,2,1]
beta4_1 <- sim.values4[,1,2]
beta4_2 <- sim.values4[,2,2]
trace.x4 <- length(alpha4_1)

# Produce plot for 'converged' section
df4 <- data.frame(x4_1 = 1000:trace.x4, 
                  a4_1 = alpha4_1[1000:trace.x4],
                  a4_2 = alpha4_2[1000:trace.x4],
                  b4_1 = beta4_1[1000:trace.x4],
                  b4_2 = beta4_2[1000:trace.x4])

p4_1 <- ggplot(data = df4, aes(x = x4_1, y = a4_1)) +
  geom_line(colour = '#0093af') +
  xlab('Iteration') + ylab('Value')  +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14))

# density plot for converged section (can put this straight after)
p4_2 <- ggplot(data = df4, aes(y = a4_1)) +
  geom_density(colour = '#aa0078', fill = '#aa0078', alpha = 0.3) +
  ylab('') + xlab('density (posterior)') +
  theme(axis.title = element_text(size = 18),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_blank())

multiplot(p4_1, p4_2, cols = 2)

p4_1
p4_2

# ```

# ```{r}

# produce traceplot of burnin - just one chain

df4_2 <- data.frame(x = 1:1000, a4_2 = alpha4_2[1:1000], a4_1 = alpha4_1[1:1000])
ggplot(data = df4_2, aes(x = x, y = a4_2)) +
  geom_line(colour = '#0093af', size = 0.8) +
  geom_line(aes(x = x, y = a4_1), colour = 'grey31', size = 0.8) +
  xlab('Iteration') + ylab('Value')  +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size = 14)) +
  annotate("rect", xmin = 0, xmax = 375, ymin = -4, ymax = 5, 
           alpha = 0.2, fill = "#aa0078")

# ```

# Look the Gelman - Rubin diagnostic

# ```{r}

# assess convergence - look at the initial behaviour of 3 chains
set.seed(1001)
#set.seed(Sys.time())

inits4_3 <- list(alpha4 = -15, beta4 = 10) 
inits4_4 <- list(alpha4 = 10, beta4 = -10)
inits4_5 <- list(alpha4 = -5, beta4 = -20) 

jags.inits4_1 <- list(inits4_3, inits4_4, inits4_5)

#jags.param <- c("alpha")
jags.param4_1 <- c("alpha4", "beta4")

jags.mod.fit4_1 <- jags(data = jags.data4, inits = jags.inits4_1,
                     parameters.to.save = jags.param4_1, n.thin = 1,
                     n.chain = 3, n.burnin = 0, DIC = FALSE,
                     n.iter = 400, model.file = jags.mod4)

sim.values4_1 <- jags.mod.fit4_1$BUGSoutput$sims.array

alpha4_3 <- sim.values4_1[,1,1]
alpha4_4 <- sim.values4_1[,2,1]
alpha4_5 <- sim.values4_1[,3,1]

# ```

# ```{r}
df4_3 <- data.frame(x = 1:200, 
                    a4_4 = alpha4_4[1:200],
                    a4_3 = alpha4_3[1:200],
                    a4_5 = alpha4_5[1:200])
ggplot(data = df4_3, aes(x = x, y = a4_4)) +
  geom_line(colour = '#0093af', size = 0.8) +
  geom_line(aes(x = x, y = a4_3), colour = 'grey31', size = 0.8) +
  geom_line(aes(x = x, y = a4_5), colour = '#aa0078', size = 0.8) +
  xlab('Iteration') + ylab('Value')  +
  theme(axis.title = element_text(size = 18),
        axis.text = element_text(size=14))

jagsfit.mcmc4_1 <- as.mcmc(jags.mod.fit4_1)
# ```


# ```{r}
### produce some Gelman-Rubin diagnostic plots


gelman.diag(jagsfit.mcmc4_1)
# Potential scale reduction factors:
#           Point est. Upper C.I.
# alpha4       1.16       1.48
# beta4        1.08       1.23

# Multivariate psrf
# 1.13

gelman.plot(jagsfit.mcmc4_1)

# ```

# Below is some additional stretch material looking at the properties of the chain when 
# the specification is centred and not - centred. Practice going through this as part of 
# going through your lecture notes. 

# ```{r}
### Dose-Response model - output of centered analysis


set.seed(973)

# Data
# dose
x5 <- c(1.69,1.72,1.75,1.78,1.81,1.83,1.86,1.88)
# number of subjects
n5 <- c(59,60,62,56,63,59,62,60)
# number of responders
r5 <- c(6,13,18,28,52,53,61,60)

N5 <- length(x5)

# Define model
jags.mod5 <- function(){
  # likelihood
  for(i in 1:N5){
    r5[i] ~ dbin(p5[i], n5[i])
    #    logit(p5[i]) <- alpha5 + beta5*(x5[i]-mean(x5))
    p5[i] <- 1/(1 + exp(-(alpha5 + beta5*(x5[i]-mean(x5)))))
  }
  # priors
  alpha5 ~ dnorm(0, 1/10000)
  beta5 ~ dnorm(0, 1/10000)
}      

# initial values for 2 chains
inits5_1 <- list(alpha5 = -5, beta5 = 10) 
inits5_2 <- list(alpha5 = 5, beta5 = -10)

jags.inits5 <- list(inits5_1, inits5_2)

# parameters to monitor
jags.param5 <- c("alpha5", "beta5")

# data
jags.data5 <- list('x5','r5','n5','N5')

# fit model
jags.mod.fit5 <- jags(data = jags.data5, inits = jags.inits5,
                     parameters.to.save = jags.param5, n.thin = 1,
                     n.chain = 2, n.burnin = 1000, DIC = FALSE,
                     n.iter = 10000, model.file = jags.mod5)

# look at summary
print(jags.mod.fit5)

# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model48377a289e0.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 1000 discarded)
# n.sims = 18000 iterations saved
#         mu.vect sd.vect  2.5%   25%    50%    75%  97.5%  Rhat n.eff
# alpha5   0.765   0.139  0.499  0.669  0.764  0.858  1.044 1.001  9700
# beta5   34.690   2.895 29.298 32.660 34.612 36.610 40.583 1.002  2100


# convert to mcmc object
jagsfit.mcmc5 <- as.mcmc(jags.mod.fit5)

summary(jagsfit.mcmc5)
# Iterations = 1001:10000
# Thinning interval = 1 
# Number of chains = 2 
# Sample size per chain = 9000 

# 1. Empirical mean and standard deviation for each variable, plus standard error of the mean:
  
#         Mean     SD     Naive SE  Time-series SE
# alpha5  0.7654  0.1389  0.001035       0.001486
# beta5  34.6902  2.8952  0.021579       0.030412

# 2. Quantiles for each variable:
  
#         2.5%     25%     50%     75%    97.5%
# alpha5  0.4991  0.6692  0.7639  0.8577  1.044
# beta5  29.2984 32.6599 34.6121 36.6096 40.583


traceplot(jagsfit.mcmc5, parms = c("alpha5", "beta5"))

# this works with the bug as well
MCMCtrace(jagsfit.mcmc5,
          ind = TRUE, 
          pdf = FALSE,
          col_den = c("#aa0078", "#0093af"))

gelman.diag(jagsfit.mcmc5)
# Potential scale reduction factors:
  
#             Point est. Upper C.I.
# alpha5          1          1
# beta5           1          1

# Multivariate psrf
# 1

gelman.plot(jagsfit.mcmc5)


# Produce plots manually to keep colours consistent
# But we can produce traceplot and density plot manually
sim.values5 <- jags.mod.fit5$BUGSoutput$sims.array

alpha5_1 <- sim.values5[,1,1]
alpha5_2 <- sim.values5[,2,1]
beta5_1 <- sim.values5[,1,2]
beta5_2 <- sim.values5[,2,2]
trace.x5 <- length(alpha5_1)

df5 <- data.frame(x = 1000:trace.x5, 
                  a5_1 = alpha5_1[1000:trace.x5],
                  a5_2 = alpha5_2[1000:trace.x5],
                  b5_1 = beta5_1[1000:trace.x5], 
                  b5_2 = beta5_2[1000:trace.x5])
p5_1 <- ggplot(data = df5) +
  geom_line(aes(x = x, y = a5_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = a5_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value') +
  ggtitle('Trace - alpha')

# density plot for converged section (can put this straight after)
p5_2 <- ggplot(data = df5) +
  geom_density(aes(x = a5_1), colour = '#aa0078') +
  geom_density(aes(x = a5_2), colour = '#0093af')+
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - alpha')

p5_3 <- ggplot(data = df5) +
  geom_line(aes(x = x, y = b5_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = b5_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value')  +
  ggtitle('Trace - beta')

# density plot for converged section (can put this straight after)
p5_4 <- ggplot(data = df5) +
  geom_density(aes(x = b5_1), colour = '#aa0078') +
  geom_density(aes(x = b5_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - beta')

multiplot(p5_1, p5_3, p5_2, p5_4, cols = 2)

#####################################################
#####################################################
#####################################################

# ```

# ```{r}
### Uncentered example

set.seed(9731)
# Data
# dose
x6 <- c(1.69,1.72,1.75,1.78,1.81,1.83,1.86,1.88)
# number of subjects
n6 <- c(59,60,62,56,63,59,62,60)
# number of responders
r6 <- c(6,13,18,28,52,53,61,60)

N6 <- length(x6)

# Define model
jags.mod6 <- function(){
  # likelihood
  for(i in 1:N6){
    r6[i] ~ dbin(p6[i], n6[i])
    #    logit(p6[i]) <- alpha6 + beta6*(x6[i]-mean(x6))
    p6[i] <-1/(1 + exp(-(alpha6 + beta6*(x6[i]))))
    #    p6[i] <- exp(alpha6 + beta6*x6[i])/(1 + exp(alpha6 + beta6*x6[i]))
  }
  # priors
  #  alpha6 ~ dunif(-1000,1000)
  #  beta6 ~ dunif(-1000,1000)
  alpha6 ~ dnorm(0,1/10000)
  beta6 ~ dnorm(0,1/10000)
}      


# initial values for 2 chains
inits6_1 <- list(alpha6 = -5, beta6 = 10) 
inits6_2 <- list(alpha6 = 5, beta6 = -10)

jags.inits6 <- list(inits6_1, inits6_2)

# parameters to monitor
jags.param6 <- c("alpha6","beta6")

# data
jags.data6 <- list('x6','r6','n6','N6')

# fit model
jags.mod.fit6 <- jags(data = jags.data6,inits = jags.inits6,
                     parameters.to.save = jags.param6, n.thin = 1,
                     n.chain = 2, n.burnin = 500, DIC = FALSE,
                     n.iter = 10000, model.file = jags.mod6)

# look at summary
print(jags.mod.fit6)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model483443d17f1.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 500 discarded)
# n.sims = 19000 iterations saved
#         mu.vect sd.vect  2.5%     25%     50%     75%   97.5%  Rhat   n.eff
# alpha6 -48.456  12.231 -62.530 -58.093 -50.538 -43.328 -15.092 1.148    18
# beta6   27.445   6.877   8.701  24.564  28.611  32.867  35.367 1.210    19


# convert to mcmc object
jagsfit.mcmc6 <- as.mcmc(jags.mod.fit6)

traceplot(jagsfit.mcmc6, parms = c("alpha6", "beta6"))

# this works with the bug as well
MCMCtrace(jagsfit.mcmc6,
          type = 'trace',
          ind = TRUE, 
          pdf = FALSE)

gelman.diag(jagsfit.mcmc6)
# Potential scale reduction factors:
#           Point est. Upper C.I.
# alpha6       1.55       4.48
# beta6        1.55       4.47

# Multivariate psrf
# 1.29

gelman.plot(jagsfit.mcmc6)


sim.values6 <- jags.mod.fit6$BUGSoutput$sims.array

alpha6_1 <- sim.values6[,1,1]
alpha6_2 <- sim.values6[,2,1]
beta6_1 <- sim.values6[,1,2]
beta6_2 <- sim.values6[,2,2]
trace.x6 <- length(alpha6_1)

df6 <- data.frame(x = 1000:trace.x6, 
                  a6_1 = alpha6_1[1000:trace.x6],
                  a6_2 = alpha6_2[1000:trace.x6],
                  b6_1 = beta6_1[1000:trace.x6],
                  b6_2 = beta6_2[1000:trace.x6])
p6_1 <- ggplot(data = df6) +
  geom_line(aes(x = x, y = a6_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = a6_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value')  +
  ggtitle('Trace - alpha')

# density plot for converged section (can put this straight after)
p6_2 <- ggplot(data = df6) +
  geom_density(aes(x = a6_1), colour = '#aa0078') +
  geom_density(aes(x = a6_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - alpha')

p6_3 <- ggplot(data = df6) +
  geom_line(aes(x = x, y = b6_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = b6_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value') +
  ggtitle('Trace - beta')

# density plot for converged section (can put this straight after)
p6_4 <- ggplot(data = df6) +
  geom_density(aes(x = b6_1), colour = '#aa0078') +
  geom_density(aes(x = b6_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - beta')

multiplot(p6_1, p6_3, p6_2, p6_4, cols = 2)


#####################################################
#####################################################
#####################################################

# ```

# ```{r}
### Standardised example

set.seed(9731)
# Data
# dose
x7 <- c(1.69,1.72,1.75,1.78,1.81,1.83,1.86,1.88)
# number of subjects
n7 <- c(59,60,62,56,63,59,62,60)
# number of responders
r7 <- c(6,13,18,28,52,53,61,60)

N7 <- length(x7)

# Define model
jags.mod7 <- function(){
  # likelihood
  for(i in 1:N7){
    r7[i] ~ dbin(p7[i], n7[i])
    #    logit(p7[i]) <- alpha7 + beta7*((x7[i]-mean(x7))/sd(x7))
    p7[i] <- 1/(1 + exp(-(alpha7 + beta7*((x7[i]-mean(x7))/sd(x7)))))
  }
  # priors
  #  alpha7 ~ dunif(-1000,1000)
  #  beta7 ~ dunif(-1000,1000)
  alpha7 ~ dnorm(0, 1/10000)
  beta7 ~ dnorm(0, 1/10000)
}      


# initial values for 2 chains
inits7_1 <- list(alpha7 = -5, beta7 = 10) 
inits7_2 <- list(alpha7 = 5, beta7 = -10)

jags.inits7 <- list(inits7_1, inits7_2)

# parameters to monitor
jags.param7 <- c("alpha7","beta7")

# data
jags.data7 <- list('x7','r7','n7','N7')

# fit model
jags.mod.fit7 <- jags(data = jags.data7,inits = jags.inits7,
                      parameters.to.save = jags.param7, n.thin = 1,
                      n.chain = 2, n.burnin = 500, DIC = FALSE,
                      n.iter = 10000, model.file = jags.mod7)

# look at summary
print(jags.mod.fit7)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model4836eef237.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 500 discarded)
# n.sims = 19000 iterations saved
#         mu.vect sd.vect  2.5%   25%   50%   75% 97.5%  Rhat n.eff
# alpha7   0.767   0.140  0.499 0.673 0.765 0.861 1.048 1.001 16000
# beta7    2.333   0.193  1.972 2.201 2.329 2.460 2.728 1.001  8500


# convert to mcmc object
jagsfit.mcmc7 <- as.mcmc(jags.mod.fit7)

traceplot(jagsfit.mcmc7, parms = c("alpha7", "beta7"))

# this works with the bug as well
MCMCtrace(jagsfit.mcmc7,
          type = 'trace',
          ind = TRUE, 
          pdf = FALSE)

gelman.diag(jagsfit.mcmc7)
# Potential scale reduction factors:
#           Point est. Upper C.I.
# alpha7          1          1
# beta7           1          1

# Multivariate psrf
# 1

gelman.plot(jagsfit.mcmc7)


sim.values7 <- jags.mod.fit7$BUGSoutput$sims.array

alpha7_1 <- sim.values7[,1,1]
alpha7_2 <- sim.values7[,2,1]
beta7_1 <- sim.values7[,1,2]
beta7_2 <- sim.values7[,2,2]
trace.x7 <- length(alpha7_1)

df7 <- data.frame(x = 1000:trace.x7, 
                  a7_1 = alpha7_1[1000:trace.x7],
                  a7_2 = alpha7_2[1000:trace.x7],
                  b7_1 = beta7_1[1000:trace.x7],
                  b7_2 = beta7_2[1000:trace.x7])
p7_1 <- ggplot(data = df7) +
  geom_line(aes(x = x, y = a7_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = a7_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value')  +
  ggtitle('Trace - alpha')

# density plot for converged section (can put this straight after)
p7_2 <- ggplot(data = df7) +
  geom_density(aes(x = a7_1), colour = '#aa0078') +
  geom_density(aes(x = a7_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - alpha')

p7_3 <- ggplot(data = df7) +
  geom_line(aes(x = x, y = b7_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = b7_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value') +
  ggtitle('Trace - beta')

# density plot for converged section (can put this straight after)
p7_4 <- ggplot(data = df7) +
  geom_density(aes(x = b7_1), colour = '#aa0078') +
  geom_density(aes(x = b7_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - beta')

multiplot(p7_1, p7_3, p7_2, p7_4, cols = 2)


# ```

#####################################################
#####################################################
#####################################################

# ```{r}
### Root of the quadratic centred example

set.seed(9731)
# Data
# dose
x8 <- c(1.69,1.72,1.75,1.78,1.81,1.83,1.86,1.88)
# number of subjects
n8 <- c(59,60,62,56,63,59,62,60)
# number of responders
r8 <- c(6,13,18,28,52,53,61,60)

N8 <- length(x8)

# Define model
jags.mod8 <- function(){
  # likelihood
  for(i in 1:N8){
    r8[i] ~ dbin(p8[i], n8[i])
    #    logit(p8[i]) <- alpha8 + beta8*(sqrt((x8[i]-mean(x8))^2)
    p8[i] <- 1/(1 + exp(-(alpha8 + beta8*(sqrt((x8[i]-mean(x8))^2)))))
  }
  # priors
  #  alpha8 ~ dunif(-1000,1000)
  #  beta8 ~ dunif(-1000,1000)
  alpha8 ~ dnorm(0, 1/10000)
  beta8 ~ dnorm(0, 1/10000)
}      


# initial values for 2 chains
inits8_1 <- list(alpha8 = -5, beta8 = 10) 
inits8_2 <- list(alpha8 = 5, beta8 = -10)

jags.inits8 <- list(inits8_1, inits8_2)

# parameters to monitor
jags.param8 <- c("alpha8","beta8")

# data
jags.data8 <- list('x8','r8','n8','N8')

# fit model
jags.mod.fit8 <- jags(data = jags.data8,inits = jags.inits8,
                      parameters.to.save = jags.param8, n.thin = 1,
                      n.chain = 2, n.burnin = 500, DIC = FALSE,
                      n.iter = 10000, model.file = jags.mod8)

# look at summary
print(jags.mod.fit8)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model483109f6552.txt", fit using jags,
# 2 chains, each with 10000 iterations (first 500 discarded)
# n.sims = 19000 iterations saved
#         mu.vect sd.vect   2.5%    25%    50%   75%  97.5%  Rhat n.eff
# alpha8   0.799   0.196   0.423  0.666  0.798  0.93  1.186 1.001 19000
# beta8   -6.656   3.054 -12.790 -8.742 -6.612 -4.60 -0.817 1.001 19000


# convert to mcmc object
jagsfit.mcmc8 <- as.mcmc(jags.mod.fit8)

traceplot(jagsfit.mcmc8, parms = c("alpha8", "beta8"))

# this works with the bug as well
MCMCtrace(jagsfit.mcmc8,
          type = 'trace',
          ind = TRUE, 
          pdf = FALSE)

gelman.diag(jagsfit.mcmc8)
# Potential scale reduction factors:
#           Point est. Upper C.I.
# alpha8          1          1
# beta8           1          1

# Multivariate psrf
# 1

gelman.plot(jagsfit.mcmc8)


sim.values8 <- jags.mod.fit8$BUGSoutput$sims.array

alpha8_1 <- sim.values8[,1,1]
alpha8_2 <- sim.values8[,2,1]
beta8_1 <- sim.values8[,1,2]
beta8_2 <- sim.values8[,2,2]
trace.x8 <- length(alpha8_1)

df8 <- data.frame(x = 1000:trace.x8, 
                  a8_1 = alpha8_1[1000:trace.x8],
                  a8_2 = alpha8_2[1000:trace.x8],
                  b8_1 = beta8_1[1000:trace.x8],
                  b8_2 = beta8_2[1000:trace.x8])
p8_1 <- ggplot(data = df8) +
  geom_line(aes(x = x, y = a8_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = a8_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value')  +
  ggtitle('Trace - alpha')

# density plot for converged section (can put this straight after)
p8_2 <- ggplot(data = df8) +
  geom_density(aes(x = a8_1), colour = '#aa0078') +
  geom_density(aes(x = a8_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - alpha')

p8_3 <- ggplot(data = df8) +
  geom_line(aes(x = x, y = b8_1), colour = '#aa0078', alpha = 0.5) +
  geom_line(aes(x = x, y = b8_2), colour = '#0093af', alpha = 0.5) +
  xlab('Iteration') + ylab('Value') +
  ggtitle('Trace - beta')

# density plot for converged section (can put this straight after)
p8_4 <- ggplot(data = df8) +
  geom_density(aes(x = b8_1), colour = '#aa0078') +
  geom_density(aes(x = b8_2), colour = '#0093af') +
  ylab('') + xlab('Parameter estimate') +
  ggtitle('Density - beta')

multiplot(p8_1, p8_3, p8_2, p8_4, cols = 2)


# ```


