rm(list=ls())
library(Rcpp)
library(RcppArmadillo)
library(fastGHQuad)

Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
sourceCpp("BiLogit_Intercept.cpp")
sourceCpp("Poisson_Intercept.cpp")

###############################################################################################################
###############################################################################################################
#binary logit model with random intercept
###############################################################################################################
set.seed(548)
n = 100 #number of subjects (clusters)
d = 5 #cluster size
Y = matrix(rep(0,n*d), nrow = n, ncol = d)
alpha = 0.6
sigma = 1.5
for (i in 1:n){
  v_i = rnorm(n = 1,mean = 0,sd = sigma)
  Y[i,] = rbinom(n = d, size = 1, prob = exp(alpha+v_i)/(1+exp(alpha+v_i)))
}

#parameters for Quasi-Newton method
v0 = 0;
x0 = c(0.5, 1);
t0 = 0.1
tc = 0.1
deltadiff = 10^-8
tol_v = 10^-8
tol_x = 10^-8

#Laplace approximation
# Logit_output_LA = BiLogitLAMLE(Y = Y, v0 = v0, x0 = x0, t0 = t0, tc = tc,  
#              tol_v = tol_v, tol_x = tol_x)
z = gaussHermiteData(1)$x
w = 1
Logit_output_LA = BiLogitAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w,
                                  t0 = t0, tc = tc, deltadiff = deltadiff, 
                                  tol_v = tol_v, tol_x = tol_x)

#Adaptive Gaussian-Hermite quadrature
#nq:5
z = gaussHermiteData(5)$x
w = gaussHermiteData(5)$w
Logit_output_AGH5 = BiLogitAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w,
                            t0 = t0, tc = tc, deltadiff = deltadiff, 
                           tol_v = tol_v, tol_x = tol_x)
#nq:9
z = gaussHermiteData(9)$x
w = gaussHermiteData(9)$w
Logit_output_AGH9 = BiLogitAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w,
                            t0 = t0, tc = tc, deltadiff = deltadiff, 
                            tol_v = tol_v, tol_x = tol_x)
# cbind(Logit_output_LA$theta, Logit_output_LA$H)
# cbind(Logit_output_AGH5$theta, Logit_output_AGH5$H)
# cbind(Logit_output_AGH9$theta, Logit_output_AGH9$H)

c(Logit_output_LA$theta, Logit_output_AGH5$theta, Logit_output_AGH9$theta)



###############################################################################################################
###############################################################################################################
#Poisson regression with random intercept
###############################################################################################################
set.seed(548)
n = 100 #number of subjects (clusters)
d = 2 #cluster size
Y = matrix(rep(0,n*d), nrow = n, ncol = d)
alpha = 0
sigma = 0.5
for (i in 1:n){
  v_i = rnorm(n = 1,mean = 0,sd = sigma)
  Y[i,] = rpois(n = d, exp(v_i + alpha))
}

#parameters for Quasi-Newton method
v0 = 0;
x0 = c(-1, 0.5);
t0 = 0.1
tc = 0.1
deltadiff = 10^-8
tol_v = 10^-8
tol_x = 10^-8

#Laplace approximation
# Pois_output_LA = PoisLAMLE(Y = Y, v0 = v0, x0 = x0, t0 = t0, tc = tc,  
#                          tol_v = tol_v, tol_x = tol_x)
z = gaussHermiteData(1)$x
w = 1
Pois_output_LA = PoisAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w,
                              t0 = t0, tc = tc, deltadiff = deltadiff,
                              tol_v = tol_v, tol_x = tol_x)
#Adaptive Gaussian-Hermite quadrature
#nq:5
z = gaussHermiteData(5)$x
w = gaussHermiteData(5)$w
Pois_output_AGH5 = PoisAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w,
                            t0 = t0, tc = tc, deltadiff = deltadiff,
                            tol_v = tol_v, tol_x = tol_x)
#nq:9
z = gaussHermiteData(9)$x
w = gaussHermiteData(9)$w
Pois_output_AGH9 = PoisAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w,
                            t0 = t0, tc = tc, deltadiff = deltadiff,
                            tol_v = tol_v, tol_x = tol_x)
# cbind(Pois_output_LA$theta, Pois_output_LA$H)
# cbind(Pois_output_AGH5$theta, Pois_output_AGH5$H)
# cbind(Pois_output_AGH9$theta, Pois_output_AGH9$H)
output = c(Pois_output_LA$theta, Pois_output_AGH5$theta, Pois_output_AGH9$theta)
output

Sys.setenv(JAVA_HOME='C:/Program Files/Java/jdk1.8.0_131/jre')
library(XLConnect)
library(xlsx)
wb <- XLConnect::loadWorkbook("tmp.xlsx", create = TRUE)
sheetname="Sheet1"
XLConnect::writeWorksheet(wb,t(round(output,digits=3)),sheetname,startRow = 1, startCol = 1, header = FALSE)
XLConnect::saveWorkbook(wb)