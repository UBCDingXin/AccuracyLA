rm(list=ls())
library(Rcpp)
library(RcppArmadillo)
library(fastGHQuad)
library(gtools)
library(MASS)

Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
sourceCpp("Poisson_AR1.cpp")
sourceCpp("BiLogit_AR1.cpp")


###############################################################################################################
###############################################################################################################
#Poisson regression
###############################################################################################################
set.seed(10)
n = 100 #number of subjects (clusters)
d = 3 #cluster size; also equals to number of random effects r
Y = matrix(rep(0,n*d), nrow = n, ncol = d)
alpha = 0
rho = 0.4
sigma = 0.5

Sigma = array(0,c(d,d))
for (i in 1:d){
  for (j in 1:d){
    Sigma[i,j] = sigma^2 * rho^abs(i-j);
  }
}

for (i in 1:n){
  v_i = mvrnorm(n = 1, mu = rep(0,d), Sigma = Sigma)
  Y[i,] = rpois(n = d, exp(v_i + alpha))
}

#parameters for Quasi-Newton method
v0 = 0;
x0 = c(alpha, rho, sigma);
t0 = 0.5
tc = 0.5
deltadiff = 10^-8
tol_v = 10^-5
tol_x = 10^-5


#Laplace approximation
nq = 1
x = 0:(nq-1)
comb_indx = t(permutations(nq, d, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = 1
Pois_output_LA = PoisAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w, comb_indx = comb_indx,
                            t0 = t0, tc = tc, deltadiff = deltadiff,
                            tol_v = tol_v, tol_x = tol_x)
#Adaptive Gaussian-Hermite quadrature
#nq:5
nq = 5
x = 0:(nq-1)
comb_indx = t(permutations(nq, d, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Pois_output_AGH5 = PoisAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w, comb_indx = comb_indx,
                              t0 = t0, tc = tc, deltadiff = deltadiff,
                              tol_v = tol_v, tol_x = tol_x)
#nq:9
nq = 9
x = 0:(nq-1)
comb_indx = t(permutations(nq, d, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Pois_output_AGH9 = PoisAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w, comb_indx = comb_indx,
                              t0 = t0, tc = tc, deltadiff = deltadiff,
                              tol_v = tol_v, tol_x = tol_x)

output = c(Pois_output_LA$theta, Pois_output_AGH5$theta, Pois_output_AGH9$theta)
output

# cbind(Pois_output_LA$theta, Pois_output_LA$H)
# cbind(Pois_output_AGH5$theta, Pois_output_AGH5$H)
# cbind(Pois_output_AGH9$theta, Pois_output_AGH9$H)

Sys.setenv(JAVA_HOME='C:/Program Files/Java/jdk1.8.0_131/jre')
library(XLConnect)
library(xlsx)
wb <- XLConnect::loadWorkbook("tmp.xlsx", create = TRUE)
sheetname="Sheet1"
XLConnect::writeWorksheet(wb,t(round(output,digits=3)),sheetname,startRow = 1, startCol = 1, header = FALSE)
XLConnect::saveWorkbook(wb)


###############################################################################################################
###############################################################################################################
#Logit regression
###############################################################################################################
set.seed(1000)
n = 100 #number of subjects (clusters)
d = 2 #cluster size; also equals to number of random effects r
Y = matrix(rep(0,n*d), nrow = n, ncol = d)
alpha = 0.6
rho = 0.4
sigma = 1

Sigma = array(0,c(d,d))
for (i in 1:d){
  for (j in 1:d){
    Sigma[i,j] = sigma^2 * rho^abs(i-j);
  }
}

for (i in 1:n){
  v_i = mvrnorm(n = 1, mu = rep(0,d), Sigma = Sigma)
  Y[i,] = rbinom(n = d, size = 1, prob = exp(alpha+v_i)/(1+exp(alpha+v_i)))
}

#parameters for Quasi-Newton method
v0 = 0;
x0 = c(0.6, 0.4, 1);
t0 = 0.5
tc = 0.5
deltadiff = 10^-8
tol_v = 10^-5
tol_x = 10^-5


#Laplace approximation
nq = 1
x = 0:(nq-1)
comb_indx = t(permutations(nq, d, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = 1
Logit_output_LA = LogitAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w, comb_indx = comb_indx,
                              t0 = t0, tc = tc, deltadiff = deltadiff,
                              tol_v = tol_v, tol_x = tol_x)
#Adaptive Gaussian-Hermite quadrature
#nq:5
nq = 5
x = 0:(nq-1)
comb_indx = t(permutations(nq, d, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Logit_output_AGH5 = LogitAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w, comb_indx = comb_indx,
                                t0 = t0, tc = tc, deltadiff = deltadiff,
                                tol_v = tol_v, tol_x = tol_x)
#nq:9
nq = 9
x = 0:(nq-1)
comb_indx = t(permutations(nq, d, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Logit_output_AGH9 = LogitAGHMLE(Y = Y, v0 = v0, x0 = x0, z = z, w = w, comb_indx = comb_indx,
                                t0 = t0, tc = tc, deltadiff = deltadiff,
                                tol_v = tol_v, tol_x = tol_x)

output = c(Logit_output_LA$theta, Logit_output_AGH5$theta, Logit_output_AGH9$theta)
output

Sys.setenv(JAVA_HOME='C:/Program Files/Java/jdk1.8.0_131/jre')
library(XLConnect)
library(xlsx)
wb <- XLConnect::loadWorkbook("tmp.xlsx", create = TRUE)
sheetname="Sheet1"
XLConnect::writeWorksheet(wb,t(round(output,digits=3)),sheetname,startRow = 1, startCol = 1, header = FALSE)
XLConnect::saveWorkbook(wb)