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


# Task B

# 2) The Dugongs example from Lecture 4

# ```{r}

### Dugong

set.seed(123)

x9 <- c( 1.0,  1.5,  1.5,  1.5, 2.5,   4.0,  5.0,  5.0,  7.0,
        8.0,  8.5,  9.0,  9.5, 9.5,  10.0, 12.0, 12.0, 13.0,
        13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5)

Y9 = c(1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
      2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43,
      2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57)

N9 = 27
# ```


# ```{r}
jags.data9 <- list("x9", "Y9", "N9")
# ```


# ```{r}
jags.mod9 <- function(){
  for (i in 1:N9) {
    Y9[i] ~ dnorm(mu9[i], tau9)
    mu9[i] <- alpha9 - beta9*gamma9^x9[i] 
  }
  # priors
  alpha9 ~ dunif(0, 100)
  beta9 ~ dunif(0, 100)
  gamma9 ~ dunif(0, 1.0)
  
  tau9 ~ dgamma(0.001, 0.001)
  sigma9 <- 1/sqrt(tau9)
}
# ```


# ```{r}
# parameters we want to monitor
jags.param9 <- c("mu9","alpha9","beta9","gamma9","sigma9")

# Initial values
inits9_1 <- list(alpha9 = 1, beta9 = 1, tau9 = 1, gamma9 = 0.9)
inits9_2 <- list(alpha9 = 10, beta9 = 3, tau9 = 5, gamma9 = 0.1)
jags.inits9 <- list(inits9_1, inits9_2)
# ```


# ```{r}
# Fit JAGS
jags.mod.fit9 <- jags(data = jags.data9, inits = jags.inits9,
                     parameters.to.save = jags.param9, n.chains = 2, 
                     n.iter = 20000, n.burnin = 10000, n.thin = 1, 
                     model.file = jags.mod9)
# ```

print(jags.mod.fit9)
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model48317beb501.txt", fit using jags,
# 2 chains, each with 20000 iterations (first 10000 discarded)
# n.sims = 20000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# alpha9     2.654   0.073   2.531   2.605   2.646   2.694   2.818 1.001  6300
# beta9      0.974   0.077   0.826   0.923   0.972   1.021   1.130 1.001 20000
# gamma9     0.863   0.033   0.788   0.845   0.866   0.885   0.917 1.001  4700
# mu9[1]     1.814   0.054   1.704   1.779   1.815   1.850   1.916 1.001  3700
# mu9[2]     1.873   0.044   1.785   1.844   1.873   1.902   1.959 1.001  4200
# mu9[3]     1.873   0.044   1.785   1.844   1.873   1.902   1.959 1.001  4200
# mu9[4]     1.873   0.044   1.785   1.844   1.873   1.902   1.959 1.001  4200
# mu9[5]     1.979   0.033   1.912   1.957   1.979   2.000   2.046 1.001  9300
# mu9[6]     2.109   0.032   2.052   2.088   2.108   2.129   2.174 1.001 20000
# mu9[7]     2.181   0.033   2.123   2.160   2.180   2.201   2.251 1.001 20000
# mu9[8]     2.181   0.033   2.123   2.160   2.180   2.201   2.251 1.001 20000
# mu9[9]     2.297   0.032   2.237   2.275   2.296   2.317   2.365 1.001 19000
# mu9[10]    2.343   0.031   2.283   2.322   2.342   2.363   2.407 1.001 16000
# mu9[11]    2.363   0.030   2.305   2.343   2.363   2.383   2.426 1.001 15000
# mu9[12]    2.383   0.029   2.325   2.363   2.383   2.402   2.442 1.001 15000
# mu9[13]    2.401   0.028   2.344   2.382   2.401   2.419   2.457 1.001 16000
# mu9[14]    2.401   0.028   2.344   2.382   2.401   2.419   2.457 1.001 16000
# mu9[15]    2.417   0.027   2.362   2.399   2.417   2.435   2.471 1.001 17000
# mu9[16]    2.473   0.024   2.424   2.457   2.473   2.489   2.520 1.001 20000
# mu9[17]    2.473   0.024   2.424   2.457   2.473   2.489   2.520 1.001 20000
# mu9[18]    2.495   0.024   2.449   2.480   2.496   2.511   2.541 1.001 20000
# mu9[19]    2.495   0.024   2.449   2.480   2.496   2.511   2.541 1.001 20000
# mu9[20]    2.524   0.024   2.476   2.507   2.524   2.540   2.570 1.001 20000
# mu9[21]    2.539   0.026   2.488   2.523   2.540   2.556   2.589 1.001 20000
# mu9[22]    2.539   0.026   2.488   2.523   2.540   2.556   2.589 1.001 20000
# mu9[23]    2.553   0.027   2.499   2.535   2.554   2.571   2.607 1.001 20000
# mu9[24]    2.560   0.028   2.502   2.541   2.560   2.578   2.615 1.001 20000
# mu9[25]    2.606   0.042   2.524   2.579   2.607   2.634   2.688 1.001  7500
# mu9[26]    2.632   0.054   2.529   2.595   2.630   2.667   2.744 1.001  5700
# mu9[27]    2.638   0.058   2.530   2.598   2.635   2.674   2.759 1.001  5500
# sigma9     0.099   0.015   0.075   0.088   0.097   0.108   0.133 1.001 20000
# deviance -49.062   3.139 -52.993 -51.378 -49.761 -47.530 -41.087 1.001 20000


# ```{r}
mu9 <- jags.mod.fit9$BUGSoutput$mean$mu9
sd9 <- jags.mod.fit9$BUGSoutput$sd$mu9
# ```


# ```{r}
df9 <- data.frame(x9 = x9, y9 = Y9, mu9 = mu9, 
                  lower9 = mu9 - 1.96*sd9, upper9 = mu9 + 1.96*sd9)

ggplot(data = df9) +
  geom_point(aes(x = x9, y = y9), colour = 'grey31', size = 2) +
  geom_line(aes(x = x9, y = mu9), colour = '#0093af', size = 1) +
  geom_line(aes(x = x9, y = lower9), linetype = "dashed", colour = "#aa0078", size = 1) +
  geom_line(aes(x = x9, y = upper9), linetype = "dashed", colour = "#aa0078", size = 1) +
  xlab('Age') + ylab('Length') + ggtitle('Dugong fitted model') +
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 12),
        plot.title = element_text(size = 18))



ggplot(data = df9) +
  geom_point(aes(x = x9, y = y9), colour = '#0093af', size = 2) +
  xlab('Age') + ylab('Length') +
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 12),
        plot.title = element_text(size = 18))
# ```

# Now explore prediction
# ```{r}
## Prediction 1

x10 <- c( 1.0,  1.5,  1.5,  1.5, 2.5,   4.0,  5.0,  5.0,  7.0,
         8.0,  8.5,  9.0,  9.5, 9.5,  10.0, 12.0, 12.0, 13.0,
         13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5)

Y10 = c(1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
       2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43,
       2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57)

N10 = 27

# ```{r}
jags.data10 <- list("x10", "Y10", "N10")
# ```

jags.mod10 <- function(){
  for(i in 1:N10) {
    Y10[i] ~ dnorm(mu10[i], tau10)
    mu10[i] <- alpha10 - beta10 * gamma10^(x10[i])			
  }
  alpha10 ~ dunif(0, 100)
  beta10 ~ dunif(0, 100)
  gamma10 ~ dunif(0, 1.0)
  
  tau10 ~ dgamma(0.001, 0.001)
  sigma10 <- 1 / sqrt(tau10)
  
  mu35 <- alpha10 - beta10*gamma10^35
  mu40 <- alpha10 - beta10*gamma10^40
  y35 ~ dnorm(mu35, tau10)
  y40 ~ dnorm(mu40, tau10)
}
# ```


# ```{r}
# Note that the interval around mu40 will reflect uncertainty concerning fitted parameters.
# The interval around y40 will additionally reflect sampling error sigma and uncertainty about sigma.

# Also, we have to specify initial values for the new stochastic nodes y35 and y40.
# E.g.
inits10_1 <- list(alpha10 = 1, beta10 = 1, tau10 = 1, gamma10 = 0.9, y35 = 2.4, y40 = 2.62)
inits10_2 <- list(alpha10 = 10, beta10 = 3, tau10 = 5, gamma10 = 0.1, y35 = 2.5, y40 = 2.6)
jags.inits10 <- list(inits10_1, inits10_2)

# Monitor the parameters we are using for the prediction
jags.param10 <- c("y35","y40","mu35","mu40")
# ```


# ```{r}
# Fitting the new model
jags.mod.fit10 <- jags(data = jags.data10, inits = jags.inits10,
                     parameters.to.save = jags.param10, n.chains = 2, 
                     n.iter = 20000, n.burnin = 10000, n.thin = 1, 
                     model.file = jags.mod10)
# ```


# ```{r}
# Get point and interval estimates
print(jags.mod.fit10)
# Also, Rhat values show convergence
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model48317ec3451.txt", fit using jags,
# 2 chains, each with 20000 iterations (first 10000 discarded)
# n.sims = 20000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# mu35       2.640   0.060   2.533   2.598   2.636   2.677   2.768 1.009   230
# mu40       2.644   0.063   2.534   2.599   2.640   2.683   2.783 1.009   220
# y35        2.638   0.117   2.409   2.563   2.637   2.714   2.872 1.003   710
# y40        2.644   0.118   2.415   2.566   2.644   2.722   2.881 1.003   920
# deviance -49.110   3.071 -52.958 -51.352 -49.803 -47.597 -41.464 1.002 20000


# Plot the prediction results
# ```{r}
sim.values10 <- jags.mod.fit10$BUGSoutput$sims.list

df35 <- data.frame(mu35 = sim.values10$mu35,
                   y35 = sim.values10$y35) %>%
  gather(key = 'node', value = 'value')

df40 <- data.frame(mu40 = sim.values10$mu40,
                   y40 = sim.values10$y40) %>%
  gather(key = 'node', value = 'value')

ggplot(data = df35, aes(x = value, colour = node)) +
  geom_density(size = 1) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 20)) +
  ggtitle('Prediction at age 35') +
  scale_color_manual(labels = c('mu35','y35'), 
                     values = c("#aa0078","#0093af")) +
  theme(legend.title = element_blank(),
        legend.text = element_text(size = 14),
        legend.position = c(0.8, 0.8)) +
  geom_vline(xintercept = mean(sim.values10$mu35), 
             linetype = "dashed", colour = "grey31", size = 0.7)


ggplot(data = df40, aes(x = value, colour = node)) +
  geom_density(size = 1) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 20)) +
  ggtitle('Prediction at age 40') +
  scale_color_manual(labels = c('mu40','y40'), 
                     values = c("#aa0078","#0093af")) +
  theme(legend.title = element_blank(),
        legend.text = element_text(size = 14),
        legend.position = c(0.8, 0.8)) +
  geom_vline(xintercept = mean(sim.values10$mu40), 
             linetype = "dashed", colour = "grey31", size = 0.7)
# ```


# ```{r}
#### Dugong - prediction as missing data


x11 <- c( 1.0,  1.5,  1.5,  1.5, 2.5,   4.0,  5.0,  5.0,  7.0,
        8.0,  8.5,  9.0,  9.5, 9.5,  10.0, 12.0, 12.0, 13.0,
        13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5,35,40)

Y11 = c(1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
      2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43,
      2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57,NA,NA)

N11 = 29
# ```


# ```{r}
jags.data11 <- list("x11", "Y11", "N11")
# ```


# ```{r}
jags.mod11 <- function(){
  for (i in 1:N11) {
    Y11[i] ~ dnorm(mu11[i], tau11)
    mu11[i] <- alpha11 - beta11*gamma11^x11[i] 
  }
  # priors
  alpha11 ~ dunif(0, 100)
  beta11 ~ dunif(0, 100)
  gamma11 ~ dunif(0, 1.0)
  
  tau11 ~ dgamma(0.001, 0.001)
  sigma11 <- 1/sqrt(tau11)
}
# ```


# ```{r}
# parameters we want to monitor
jags.param11 <- c("mu11", "Y11")
# ```


# ```{r}
# Initial values
inits11_1 <- list(alpha11 = 1, beta11 = 1, tau11 = 1, gamma11 = 0.9, Y = c(rep(NA,27), 2.4, 2.62))
inits11_2 <- list(alpha11 = 10, beta11 = 3, tau11 = 5, gamma11 = 0.1, Y = c(rep(NA,27), 2.5, 2.6))
jags.inits11 <- list(inits11_1, inits11_2)
# ```


# ```{r}
# Fit JAGS
jags.mod.fit11 <- jags(data = jags.data11, inits = jags.inits11,
                     parameters.to.save = jags.param11, n.chains = 2, 
                     n.iter = 20000, n.burnin = 10000, n.thin = 1, 
                     model.file = jags.mod11)

# Note that initial values for the missing values are automatically generated by JAGS
# ```


# ```{r}
# Look at outcome - point/interval estimates
# Rhat values show convergence
print(jags.mod.fit11)
# The first 27 Y nodes are not stochastic, since we have data available but they still appear in the outcome.
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model483145ebe1c.txt", fit using jags,
# 2 chains, each with 20000 iterations (first 10000 discarded)
# n.sims = 20000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# Y11[28]    2.641   0.118   2.410   2.563   2.641   2.717   2.876 1.001  9300
# Y11[29]    2.647   0.119   2.415   2.568   2.646   2.725   2.889 1.001  5700
# mu11[28]   2.642   0.061   2.528   2.601   2.639   2.679   2.772 1.001  8600
# mu11[29]   2.646   0.065   2.528   2.603   2.642   2.685   2.788 1.001  7400
# deviance -49.054   3.188 -52.988 -51.385 -49.800 -47.506 -40.995 1.002 20000



# ```{r}
# Print model fit:
mu11 <- jags.mod.fit11$BUGSoutput$mean$mu11
sd11 <- jags.mod.fit11$BUGSoutput$sd$mu11
Ysim11 <- jags.mod.fit11$BUGSoutput$mean$Y11

df11 <- data.frame(x = x11, y = Ysim11, mu = mu11, 
                   lower11 = mu11 - 1.96*sd11, upper11 = mu11 + 1.96*sd11)

ggplot(data = df11) +
  geom_point(aes(x = x11, y = Ysim11), colour = 'grey31', size = 2) +
  geom_line(aes(x = x11, y = mu11), colour = '#0093af', size = 1) +
  geom_line(aes(x = x11, y = lower11), linetype = "dashed", colour = "#aa0078", size = 1) +
  geom_line(aes(x = x11, y = upper11), linetype = "dashed", colour = "#aa0078", size = 1) +
  xlab('Age') + ylab('Length') +
  ggtitle('Dugong fitted model with predicted values for ages 35 and 40') +
  theme(axis.title = element_text(size = 16),
        axis.text = element_text(size = 12),
        plot.title = element_text(size = 18))
# ```

#### Dugong - prediction as missing data for ages 25 and 33


x12 <- c( 1.0,  1.5,  1.5,  1.5, 2.5,   4.0,  5.0,  5.0,  7.0,
          8.0,  8.5,  9.0,  9.5, 9.5,  10.0, 12.0, 12.0, 13.0,
          13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5,25,33)

Y12 = c(1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
        2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43,
        2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57,NA,NA)

N12 = 29
# ```


# ```{r}
jags.data12 <- list("x12", "Y12", "N12")
# ```


# ```{r}
jags.mod12 <- function(){
  for (i in 1:N12) {
    Y12[i] ~ dnorm(mu12[i], tau12)
    mu12[i] <- alpha12 - beta12*gamma12^x12[i] 
  }
  # priors
  alpha12 ~ dunif(0, 100)
  beta12 ~ dunif(0, 100)
  gamma12 ~ dunif(0, 1.0)
  
  tau12 ~ dgamma(0.001, 0.001)
  sigma12 <- 1/sqrt(tau12)
}
# ```


# ```{r}
# parameters we want to monitor
jags.param12 <- c("mu12", "Y12")
# ```


# ```{r}
# Initial values
inits12_1 <- list(alpha12 = 1, beta12 = 1, tau12 = 1, gamma12 = 0.9, Y = c(rep(NA,27), 2.4, 2.62))
inits12_2 <- list(alpha12 = 10, beta12 = 3, tau12 = 5, gamma12 = 0.1, Y = c(rep(NA,27), 2.5, 2.6))
jags.inits12 <- list(inits12_1, inits12_2)
# ```


# ```{r}
# Fit JAGS
jags.mod.fit12 <- jags(data = jags.data12, inits = jags.inits12,
                       parameters.to.save = jags.param12, n.chains = 2, 
                       n.iter = 20000, n.burnin = 10000, n.thin = 1, 
                       model.file = jags.mod12)

# Note that initial values for the missing values are automatically generated by JAGS
# ```


# ```{r}
# Look at outcome - point/interval estimates
# Rhat values show convergence
print(jags.mod.fit12)
# The first 27 Y nodes are not stochastic, since we have data available but they still appear in the outcome.
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model48358132ab6.txt", fit using jags,
# 2 chains, each with 20000 iterations (first 10000 discarded)
# n.sims = 20000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# Y12[28]    2.617   0.112   2.397   2.545   2.618   2.690   2.835 1.001 20000
# Y12[29]    2.639   0.118   2.406   2.562   2.638   2.715   2.872 1.001 20000
# mu12[28]   2.617   0.048   2.522   2.586   2.617   2.649   2.712 1.001  4800
# mu12[29]   2.638   0.060   2.526   2.597   2.635   2.676   2.765 1.001  6200
# deviance -48.979   3.215 -52.967 -51.352 -49.713 -47.414 -40.695 1.001 20000


# ```{r}
# Print model fit:
mu12 <- jags.mod.fit12$BUGSoutput$mean$mu12
sd12 <- jags.mod.fit12$BUGSoutput$sd$mu12
Ysim12 <- jags.mod.fit12$BUGSoutput$mean$Y12

df12 <- data.frame(x = x12, y = Ysim12, mu = mu12, 
                   lower12 = mu12 - 1.96*sd12, upper12 = mu12 + 1.96*sd12)

ggplot(data = df12) +
  geom_point(aes(x = x12, y = Ysim12), colour = 'grey31', size = 2) +
  geom_line(aes(x = x12, y = mu12), colour = '#0093af', size = 1) +
  geom_line(aes(x = x12, y = lower12), linetype = "dashed", colour = "#aa0078", size = 1) +
  geom_line(aes(x = x12, y = upper12), linetype = "dashed", colour = "#aa0078", size = 1) +
  xlab('Age') + ylab('Length') +
  ggtitle('Dugong fitted model with predicted values for ages 25 and 33') +
  theme(axis.title = element_text(size = 12),
        axis.text = element_text(size = 12),
        plot.title = element_text(size = 14))
# ```


x13 <- c( 1.0,  1.5,  1.5,  1.5, 2.5,   4.0,  5.0,  5.0,  7.0,
          8.0,  8.5,  9.0,  9.5, 9.5,  10.0, 12.0, 12.0, 13.0,
          13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5,25,33)

Y13 = c(1.80, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47,
        2.19, 2.26, 2.40, 2.39, 2.41, 2.50, 2.32, 2.32, 2.43,
        2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.70, 2.72, 2.57,NA,NA)

N13 = 29
# ```


# ```{r}
jags.data13 <- list("x13", "Y13", "N13")
# ```


jags.mod13 <- function(){
  for(i in 1:N13) {
    Y13[i] ~ dnorm(mu13[i], tau13)
    mu13[i] <- alpha13 - beta13 * gamma13^(x13[i])			
  }
  alpha13 ~ dunif(0, 100)
  beta13 ~ dunif(0, 100)
  gamma13 ~ dunif(0, 1.0)
  
  tau13 ~ dgamma(0.001, 0.001)
  sigma13 <- 1 / sqrt(tau13)
  
  mu25 <- alpha13 - beta13*gamma13^25
  mu33 <- alpha13 - beta13*gamma13^33
  y25 ~ dnorm(mu25, tau13)
  y33 ~ dnorm(mu33, tau13)
}
# ```


# ```{r}
# Note that the interval around mu33 will reflect uncertainty concerning fitted parameters.
# The interval around y33 will additionally reflect sampling error sigma and uncertainty about sigma.

# Also, we have to specify initial values for the new stochastic nodes y25 and y33.
# E.g.
inits13_1 <- list(alpha13 = 1, beta13 = 1, tau13 = 1, gamma13 = 0.9, y25 = 2.4, y33 = 2.62)
inits13_2 <- list(alpha13 = 13, beta13 = 3, tau13 = 5, gamma13 = 0.1, y25 = 2.5, y33 = 2.6)
jags.inits13 <- list(inits13_1, inits13_2)

# Monitor the parameters we are using for the prediction
jags.param13 <- c("y25","y33","mu25","mu33")
# ```


# ```{r}
# Fitting the new model
jags.mod.fit13 <- jags(data = jags.data13, inits = jags.inits13,
                       parameters.to.save = jags.param13, n.chains = 2, 
                       n.iter = 20000, n.burnin = 10000, n.thin = 1, 
                       model.file = jags.mod13)
# ```


# ```{r}
# Get point and interval estimates
print(jags.mod.fit13)
# Also, Rhat values show convergence
# Inference for Bugs model at "/var/folders/pc/hksslngn6_56y6dc8464zthm0000gn/T//RtmpYZ6vJw/model48313983ea5.txt", fit using jags,
# 2 chains, each with 20000 iterations (first 10000 discarded)
# n.sims = 20000 iterations saved
#           mu.vect sd.vect   2.5%     25%     50%     75%   97.5%  Rhat n.eff
# mu25       2.617   0.047   2.521   2.585   2.617   2.649   2.708 1.001 20000
# mu33       2.637   0.059   2.525   2.596   2.635   2.677   2.758 1.001 20000
# y25        2.615   0.111   2.394   2.543   2.615   2.689   2.834 1.001  6500
# y33        2.638   0.117   2.407   2.562   2.638   2.714   2.868 1.001 20000
# deviance -49.037   3.108 -52.974 -51.339 -49.726 -47.492 -41.277 1.001 15000


# Plot the prediction results
# ```{r}
sim.values13 <- jags.mod.fit13$BUGSoutput$sims.list

df25 <- data.frame(mu25 = sim.values13$mu25,
                   y25 = sim.values13$y25) %>%
  gather(key = 'node', value = 'value')

df33 <- data.frame(mu33 = sim.values13$mu33,
                   y33 = sim.values13$y33) %>%
  gather(key = 'node', value = 'value')

ggplot(data = df25, aes(x = value, colour = node)) +
  geom_density(size = 1) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 20)) +
  ggtitle('Prediction at age 25') +
  scale_color_manual(labels = c('mu25','y25'), 
                     values = c("#aa0078","#0093af")) +
  theme(legend.title = element_blank(),
        legend.text = element_text(size = 14),
        legend.position = c(0.8, 0.8)) +
  geom_vline(xintercept = mean(sim.values13$mu25), 
             linetype = "dashed", colour = "grey31", size = 0.7)


ggplot(data = df33, aes(x = value, colour = node)) +
  geom_density(size = 1) +
  theme(axis.title = element_blank(),
        axis.text = element_text(size = 14),
        plot.title = element_text(size = 20)) +
  ggtitle('Prediction at age 33') +
  scale_color_manual(labels = c('mu33','y33'), 
                     values = c("#aa0078","#0093af")) +
  theme(legend.title = element_blank(),
        legend.text = element_text(size = 14),
        legend.position = c(0.8, 0.8)) +
  geom_vline(xintercept = mean(sim.values13$mu33), 
             linetype = "dashed", colour = "grey31", size = 0.7)
# ```

