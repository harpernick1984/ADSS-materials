# Install and load rjags package to allow JAGS to be used in R
install.packages("rjags")
library(rjags)
testjags()


# Attempts to install and load rjags package to allow JAGS to be used in R
devtools::install_url("http://sourceforge.net/projects/mcmc-jags/files/rjags/4/rjags_4-4.tar.gz",
                      args="--configure-args='--with-jags-include=/Users/casallas/homebrew/opt/jags/include/JAGS
                            --with-jags-lib=/Users/casallas/homebrew/opt/jags/lib'")
install.packages("runjags")
library(runjags)
install.packages('runjags', configure.args=c('--with-jags-include=/home/emontene/JAGS-4.3.0/include/JAGS 
                                             --with-jags-lib=/home/emontene/JAGS-4.3.0/lib/ 
                                             --with-jags-modules=/home/emontene/JAGS-4.3.0/lib/JAGS/modules-4 
                                             --with-jags-bin=/home/emontene/JAGS-4.3.0/bin --enable-rpath'))
install.packages("rjags")
devtools::install_url("http://sourceforge.net/projects/mcmc-jags/files/rjags/3/rjags_3-2.tar.gz",
                      args="--configure-args='--with-jags-include=/Users/casallas/homebrew/opt/jags/include/JAGS        
                                              --with-jags-lib=/Users/casallas/homebrew/opt/jags/lib'
                            ")
devtools::install_url("http://sourceforge.net/projects/mcmc-jags/files/rjags/4/rjags_4-4.tar.gz",
                      args="--configure-args='--with-jags-include=/Users/casallas/homebrew/opt/jags/include/JAGS
--with-jags-lib=/Users/casallas/homebrew/opt/jags/lib'
")

remove.packages("rjags")
remove.packages("R2jags")
remove.packages("runjags")



# title: "MTHM017 Advanced Methods Workshop  A"
# author: "John T Bruun"
# date: "20/01/2021"
# output:
# pdf_document: default
# html_document: default
# word_document: default

  
# ```{r setup, include=FALSE}
# knitr::opts_chunk$set(echo = TRUE)
# ```

# This and next week we are looking at how we implement and compute the Bayesian statistics and inference properties. 
# You will explore how we code and run the lecture examples plus more. We run this from the R package. 
# You need both RStudio and JAGS installed onto your machines. The CEMPS Digital Learning Developers can help you 
# if you have any specific install questions (MathsDLD@exeter.ac.uk).

# Bayesian computation enables us to resolve and calculate sophisticated probability problems. In some, and especially 
# text book cases, these can be worked out exactly via algebra. However for a wide range of non-regular inference 
# problems it is not always possible to find the answers analytically. Since the availability of high volume computation 
# over the last 50 years, stochastic sampling algorithms have become a useful tool. With this Bayesian techniques 
# have been developed that enable us to solve these probability questions to a very high level of accuracy.

# To explore these you should familiarise yourself with coding and running the lecture examples explained in the first week. 
# These are reproduced here in R Markdown.

# Below is the script coding. 

# To build your coding experience use this to become familiar with:
  
# a) how to use this code as is and to reproduce the lecture note results.

# b) Use the help() facility to look at the additional settings you can alter in jags(), especially the number of chains, 
# number of iterations and burn length. Examine how the standard results for the coin and drug examples vary when the 
# sampling chains become too small. Practicing this and building understanding of how to look for convergence to a stable 
# result are an essential aspect of using these advanced probability tools. 

# c) For debug and finished analysis you can 'set the seed' and let it run freely. Discuss in your breakout groups and 
# show each other the differing results you obtain when it runs freely. Alter the chain length to compare notes. 

# d) Once you are familiar with a) - c) practice altering the jags.mod() and the data values. 

# e) Discuss in your teams what you have learnt from this workshop and be prepared to share feedback towards the end of 
# the session with your co-students. Aim to rotate this to give opportunity for as many of you to explain and to learn from each other. 

# A solution html is also provided. Please aim to run this from the basic RMarkdown code and check your answers once you have attempted the above.


# 1) Firstly load in a range of libraries
# ```{r}
install.packages("R2jags")
library(R2jags)
library(MCMCvis)
library(tidyverse)
library(TSA)
library(coda)
library(ggplot2)
library(lattice)
# ```

# We focus on using rjags. As explained in the notes - this is a way to run the JAGS/BUGS functionality- from within R. 
# This technology is relatively recent, and it makes the Bayesian inference much easier to combine with other R based analyses. 
# This is state-of-the-art functionality and you are fortunate to be learning Bayes things this way.

# 2) Coin toss example from Lecture 1. a) Setting the jags.model. b) Set the seed (uncomment to fix this). 
# c) Running the jags analysis. d) Print the result output.
# ```{r}
# a) Lecture notes code

jags.mod1 <- function(){
  Y ~ dbin(0.5, 10)
  P8 <- ifelse(Y > 7, 1, 0)
}
# ```
# ```{r}
# b) set.seed(2)
set.seed(2)
# ```
# ```{r}
# c) run the analysis (here is where you change the burn and iterations)
jags.mod.fit1 <- jags(data = list(), model.file = jags.mod1,
                      parameters.to.save = c("Y","P8"), DIC = FALSE, 
                      n.chains = 10, n.iter = 10)
# ```
# ```{r}
# d) Print the jags results
jags.mod.fit1
# ```
# Diagnostics: Look at the chain output for Y. Firstly extract the chain record and also map this into a time series object. 
# Here histogram, chain time series record together with auto and partial correlation are explored. What do the acf terms help us to assess?
# ```{r}

y.diag1 <- jags.mod.fit1$BUGSoutput$sims.list$Y
y.diag.ts1 <- ts(y.diag1)

histogram(y.diag1, xlab = "Y", plot = TRUE)
ts.plot(y.diag.ts1, xlab = "Iteration", ylab = "Y")
acf(y.diag.ts1)
acf(y.diag.ts1, type = "partial")

# ```

# Diagnostics: Now, as for before, look at the chain output for P8
# ```{r}

P8.diag1 <- jags.mod.fit1$BUGSoutput$sims.list$P8
P8.diag.ts1 <- ts(P8.diag1)

histogram(P8.diag1, xlab = "Y", plot = TRUE)
ts.plot(P8.diag.ts1, xlab = "Iteration", ylab = "Y")
acf(P8.diag.ts1)
acf(P8.diag.ts1, type = "partial")

# ```

# 2) The drug assessment example from Lecture 1 and 2. Assign the model and run jags. Note how the data numbers are hard coded into this: 
# why can this be problematic in a practical sense?
# ```{r}
set.seed(2)

# Assign the beta-binomial model relationship. 
jags.mod2 <- function(){
  theta ~ dbeta(9.2, 13.8)
  y ~ dbin(theta, 20)
  P.crit <- ifelse(y >= 15, 1, 0)
}
# ```
# ```{r}
# Run the jags analysis
jags.mod.fit2 <- jags(data = list(), n.iter = 10, n.chains = 10, 
                      parameters.to.save = c("theta", "y", "P.crit"),
                      DIC = FALSE, model.file = jags.mod2)
# ```
# Print the jags results.
# ```{r}
print(jags.mod.fit2)
# ```
# Show the diagnostics (as above). Try altering the above code for differing chains and iteration lengths 
# to see how the results and diagnostics alter. Discuss this with your Breakout Group colleagues.
# ```{r}
y.diag2 <- jags.mod.fit2$BUGSoutput$sims.list$y
y.diag.ts2 <- ts(y.diag2)

histogram(y.diag2, xlab = "Y", plot = TRUE)
ts.plot(y.diag.ts2, xlab = "Iteration", ylab = "Y")
acf(y.diag.ts2)
acf(y.diag.ts2, type = "partial")

# ```
# Diagnostics for p 
# ```{r}
p.diag2 <- jags.mod.fit2$BUGSoutput$sims.list$P.crit
p.diag.ts2 <- ts(p.diag2)

histogram(p.diag2, xlab = "P Crit", plot = TRUE)
ts.plot(p.diag.ts2, xlab = "Iteration", ylab = "P Crit")
acf(p.diag.ts2)
acf(p.diag.ts2, type = "partial")

# ```
# Diagnostics for theta:
# ```{r}
theta.diag2 <- jags.mod.fit2$BUGSoutput$sims.list$theta
theta.diag.ts2 <- ts(theta.diag2)

histogram(theta.diag2, xlab = "theta", plot = TRUE)
ts.plot(theta.diag.ts2, xlab = "Iteration", ylab = "Theta")
acf(theta.diag.ts2)
acf(theta.diag.ts2, type = "partial")

# ```

# From Lecture 2, this is how to code the model more generically
# ```{r}
set.seed(2)

# model
jags.mod3 <- function(){
  theta ~ dbeta(a, b)                      # prior disribution
  y ~ dbin(theta, n)                       # sampling distribution
  y.pred ~ dbin(theta, m)                  # predictive distribution
  P.crit <- ifelse(y.pred >= ncrit, 1, 0)  # =1 if y.pred >= ncrit, 0 o/w
}
 
# ```
# The data are passed into this model via the jags() command via a list(). Explain why y = 15 here? 
# ```{r}
# data
a = 9.2 
b = 13.8
y = 15
n = 20 
m = 40
ncrit = 25

jags.data3 <- list("a", "b", "y", "n", "m", "ncrit")

# ```
# This is how we define the parameters of interest, for the jags() command.
# ```{r}
# Parameters we want to monitor 
jags.param3 <- c("theta", "y.pred", "P.crit")
# ```
# Initial data values are set with this:
# ```{r}
# Specify initial values
jags.inits3 <- function(){
  list("theta" = 0.7, "y.pred" = 20)
}
# ```
# Run jags()  Again explore differing chain numbers and iteration lengths.
# ```{r}
# Fit the JAGS model
# n.iter = number of samples to be drawn
# n.burnin = number of samples to discard (that is we won't use these in the analysis)
jags.mod.fit3 <- jags(data = jags.data3, inits = jags.inits3,
                      parameters.to.save = jags.param3, n.chain = 10,
                      n.iter = 10, model.file = jags.mod3, n.thin = 1)
# ```
# Print results. Go into the command window to advance the traceplot functions by pressing return when asked.
# ```{r}
print(jags.mod.fit3)
traceplot(jags.mod.fit3)
# ```

# ```{r} THIS DOESN'T WORK!
jags.hist(jags.mod.fit3)
# ```
# You can plot the theta interval property with this Monte Carlo Markov Chain code extension. What does MCMC do? Discuss.
# ```{r}
jagsfit.mcmc3 <- as.mcmc(jags.mod.fit3)
MCMCtrace(jagsfit.mcmc3,
          params = 'theta',
          type = 'density',
          ind = TRUE, 
          pdf = FALSE)
# ```
# Once you are comfortable with running the above. Go through the additional tasks outlined above.

