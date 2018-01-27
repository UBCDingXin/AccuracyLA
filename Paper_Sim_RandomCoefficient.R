rm(list=ls())
library(Rcpp)
library(RcppArmadillo)
library(fastGHQuad)
library(gtools)
library(MASS)

Sys.setenv("PKG_CXXFLAGS"="-std=c++11")
sourceCpp("BiLogit_RandCoefficient.cpp")
sourceCpp("Poisson_RandCoefficient.cpp")

###############################################################################################################
###############################################################################################################
#Logit regression
###############################################################################################################
set.seed(548)
n = 100 #number of subjects (clusters)
d = 3 #cluster size; also equals to number of random effects r
beta = c(0.6,0.2)
sigma = c(1,0.5)
pix = 0.4
r = length(beta)
c = length(pix)
Y = matrix(rep(0,n*d), nrow = n, ncol = d)
XCube = array(0, c(d,c,n))
Sigma = array(0,c(r,r))
diag(Sigma) = sigma^2
for (i in 1:n){
  tmp = matrix(rbinom(n = c, size = 1, prob = pix), nrow = c, ncol = 1)
  XCube[,,i] = t(rep(tmp, d))
}

for (i in 1:n){
  v_i = mvrnorm(n = 1, mu = rep(0,r), Sigma = Sigma)
  etas = crossprod( (beta+v_i), rbind( rep(1,d), t(XCube[,,i]) ) )
  probs = exp(etas) / (1+exp(etas));
  Y[i,] = rbinom(n = d, size = 1, prob = probs)
}

#parameters for Quasi-Newton method
v0 = 0;
x0 = c(beta,sigma);
t0 = 0.1
tc = 0.1
deltadiff = 10^-10
tol_v = 10^-5
tol_x = 10^-8

nq = 1
x = 0:(nq-1)
comb_indx = t(permutations(nq, r, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = 1
Logit_output_LA = LogitAGHMLE(Y, XCube, v0_num = v0, x0 = x0, 
            z, w, comb_indx,
            tol_v = tol_v, tol_x = tol_x)

nq = 5
x = 0:(nq-1)
comb_indx = t(permutations(nq, r, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Logit_output_AGH5 = LogitAGHMLE(Y, XCube, v0_num = v0, x0 = x0, 
            z, w, comb_indx,
            tol_v = tol_v, tol_x = tol_x)

nq = 9
x = 0:(nq-1)
comb_indx = t(permutations(nq, r, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Logit_output_AGH9 = LogitAGHMLE(Y, XCube, v0_num = v0, x0 = x0, 
            z, w, comb_indx,
            tol_v = tol_v, tol_x = tol_x)

output = cbind(Logit_output_LA$theta, Logit_output_AGH5$theta, Logit_output_AGH9$theta)
output


###############################################################################################################
###############################################################################################################
#Poisson regression
###############################################################################################################
set.seed(548)
n = 100 #number of subjects (clusters)
d = 3 #cluster size; also equals to number of random effects r
beta = c(-0.2,0.1,0.15)
sigma = c(0.5,0.1,0.12)
pix = c(0.4, 0.45)
r = length(beta)
c = length(pix)
Y = matrix(rep(0,n*d), nrow = n, ncol = d)
XCube = array(0, c(d,c,n))
Sigma = array(0,c(r,r))
diag(Sigma) = sigma^2
for (i in 1:n){
  tmp = matrix(rbinom(n = c, size = 1, prob = pix), nrow = c, ncol = 1)
  XCube[,,i] = t(rep(tmp, d))
}

for (i in 1:n){
  v_i = mvrnorm(n = 1, mu = rep(0,r), Sigma = Sigma)
  etas = crossprod( (beta+v_i), rbind( rep(1,d), t(XCube[,,i]) ) )
  Y[i,] = rpois(n = d, exp(etas))
}

#parameters for Quasi-Newton method
v0 = 0;
x0 = c(beta,sigma);
# x0 = rep(0,2*r)
t0 = 0.5
tc = 0.1
deltadiff = 10^-10
tol_v = 10^-5
tol_x = 10^-8

nq = 1
x = 0:(nq-1)
comb_indx = t(permutations(nq, r, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = 1
Pois_output_LA = PoisAGHMLE(Y, XCube, v0_num = v0, x0 = x0, 
                            z, w, comb_indx,
                            tol_v = tol_v, tol_x = tol_x)

nq = 5
x = 0:(nq-1)
comb_indx = t(permutations(nq, r, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Pois_output_AGH5 = PoisAGHMLE(Y, XCube, v0_num = v0, x0 = x0, 
                              z, w, comb_indx,
                              tol_v = tol_v, tol_x = tol_x)

nq = 9
x = 0:(nq-1)
comb_indx = t(permutations(nq, r, v = x, repeats.allowed=T))
z = gaussHermiteData(nq)$x
w = gaussHermiteData(nq)$w
Pois_output_AGH9 = PoisAGHMLE(Y, XCube, v0_num = v0, x0 = x0, 
                              z, w, comb_indx,
                              tol_v = tol_v, tol_x = tol_x)

output = cbind(Pois_output_LA$theta, Pois_output_AGH5$theta, Pois_output_AGH9$theta)
round(output,digits=3)


