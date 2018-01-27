//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp:plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
using namespace Rcpp;
using namespace arma;

// g(v) = -log(f(v))
double g(arma::vec y, double alpha, arma::vec v){
  int d = y.n_elem;
  vec tmp(d);
  for (int j = 0; j < d; j++){
    tmp(j) = log(1 + exp(alpha + v(j))) - ( alpha + v(j) ) * y(j);
  }
  return sum(tmp);
}//end of function g
// first order derivative of g(v) wrt v
arma::vec gdev1(arma::vec y, double alpha, arma::vec v) {
  int d = y.n_elem;
  vec output(d);
  for(int j = 0; j < d; j++){
    output(j) = exp(alpha + v(j)) / ( 1 + exp(alpha + v(j)) ) - y(j);
  }
  return output;
}
// second order derivative of g(v) wrt v
arma::mat gdev2(arma::vec y, double alpha, arma::vec v){
  int d = y.n_elem;
  mat output(d, d, fill::zeros);
  for(int j = 0; j < d; j++){
    output(j, j) = exp(alpha + v(j)) / pow(1 + exp(alpha + v(j)), 2);
  }
  return output;
}

//root of eq.(8); compute \tilde{v}
arma::vec vForLA(arma::vec v0, double alpha, double rho, double sigma, 
                 arma::vec y, double tol_v = 0.00001){
  //v0 is the initial value for random effect
  int r = v0.n_elem;
  mat Sigma(r,r);
  for (int i = 0; i < r; i++){
    for (int j = 0; j < r; j++){
      Sigma(i,j) = pow(sigma, 2) * pow(rho, abs(i-j));
    }
  }// create the covariance matrix
  // int flag = 1;
  double dis_v = 999.0;
  vec v;
  while(dis_v > tol_v){
    // std::cout<<flag<<";"<<dis_v<<std::endl;
    v = v0 - ( gdev2(y, alpha, v0) + Sigma.i() ).i() * ( gdev1(y, alpha, v0) + Sigma.i() * v0 );//update v
    // dis_v = abs(v - v0);
    dis_v = as_scalar(sum(abs(gdev1(y, alpha, v) + Sigma.i() * v)));
    v0 = v;
    // flag++;
  }//end while
  return v;
}

//Probability limit of negative log-likelihood function
double F(arma::mat Yrev, int K, int n, 
         arma::vec nobs_distinct_cases, 
         double alpha, double rho, double sigma, arma::mat v_mat,
         arma::mat comb_indx, vec z, vec w){
  // r == d in this case
  // v_mat should be a r * K matrix; comb_indx is a r * nq^r matrix
  // z is a vector of nq nodes; w is a vector of weights
  int nq = z.n_elem; //number of nodes
  int r = v_mat.n_rows; // number of random effects
  vec tmpG(pow(nq, r)), tmpApprox(K), y;
  double output, Gvalue, Wj;
  vec v, vforG, zj(r), wj(r);
  mat Rk, Sigma(r, r);
  for (int i = 0; i < r; i++){
    for (int j = 0; j < r; j++){
      Sigma(i,j) = pow(sigma, 2) * pow(rho, abs(i-j));
    }
  }// create the covariance matrix
  for (int k = 0; k < K; k++){
    v = v_mat.col(k);
    y = vectorise(Yrev.row(k), 0);
    double ratio = nobs_distinct_cases(k) / n;
    Rk = chol(gdev2(y, alpha, v) + Sigma.i());
    for (int j = 0; j < pow(nq, r); j++){
      for (int jj = 0; jj < r; jj++){
        zj(jj) = z(comb_indx(jj, j)); //get the nodes combination
        wj(jj) = w(comb_indx(jj, j)); //get the weights combination
      }//end for loop: jj from 0 to r
      Wj = as_scalar(exp(sum(pow(zj, 2))) * prod(wj));
      vforG = v + Rk.i() * zj;
      Gvalue = - g(y, alpha, vforG) - 0.5 * as_scalar(vforG.t() * Sigma.i() * vforG);
      tmpG(j) = exp(Gvalue) * Wj;
    }//end for loop: j from 0 to nq^r
    tmpApprox(k) = ratio * ( 0.5 * log(det(Sigma)) + log(det(Rk)) - log(sum(tmpG)) );
  }//end for loop: k from 0 to K
  output = sum(tmpApprox);
  return output;
}//end of negative log-likelihood function



//--------------------------------------------------------------------------
//Quasi-Newton method
// [[Rcpp::export]]
List LogitAGHMLE(arma::mat Y, double v0_num, arma::vec x0, 
                arma::vec z, arma::vec w, arma::mat comb_indx,
                double t0 = 1, double tc = 0.1, double deltadiff = 1e-10,
                double tol_v = 0.00001, double tol_x = 0.00001) {
  int n = Y.n_rows, d = Y.n_cols; //dim of Y; in this case, r = d
  vec v0(d, fill::ones);
  v0 = v0 * v0_num;
  //regard each row of Y as a distinct case
  int K = n;
  vec nobs_distinct_cases(K,fill::ones); //store the number of obs for each case
  mat Yrev = Y;
  double dis_x = 999.0, t, Fx, Fx0;
  mat v1_mat(d, K), v11_mat(d, K), v12_mat(d, K), v13_mat(d, K),
  v2_mat(d, K), v21_mat(d, K), v22_mat(d, K), v23_mat(d, K); //v1 and v2 are random intercepts; for each row in Yrev, we will get a v.
  vec x;// x = (alpha,rho,sigma)
  vec s1, s2, deltax(3), Fdev0(3), Fdev(3); // intermediate steps for Quasi Newton
  mat Hinv0 = eye<mat>(3,3), Hinv;
  int flag_step = 1;
  
  //v for each case based on x0
  for(int k = 0; k < K; k++){
    v1_mat.col(k) = vForLA(v0, x0(0), x0(1), x0(2), vectorise(Yrev.row(k), 0), tol_v);
    v11_mat.col(k) = vForLA(v0, x0(0) + deltadiff, x0(1), x0(2), vectorise(Yrev.row(k), 0), tol_v);
    v12_mat.col(k) = vForLA(v0, x0(0), x0(1) + deltadiff, x0(2), vectorise(Yrev.row(k), 0), tol_v);
    v13_mat.col(k) = vForLA(v0, x0(0), x0(1), x0(2) + deltadiff, vectorise(Yrev.row(k), 0), tol_v);
  }//end first for loop
  
  //Quasi Newton Method
  while (dis_x > tol_x){
    // std::cout<<"Step:"<<flag_step<<"; dis:"<<dis_x<<std::endl;
    Fdev0(0) = ( F(Yrev, K, n, nobs_distinct_cases, x0(0) + deltadiff, x0(1), x0(2), v11_mat, comb_indx, z, w)
                   - F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), x0(2), v1_mat, comb_indx, z, w) ) / deltadiff;
    Fdev0(1) = ( F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1) + deltadiff, x0(2), v12_mat, comb_indx, z, w)
                   - F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), x0(2), v1_mat, comb_indx, z, w) ) / deltadiff;
    Fdev0(2) = ( F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), x0(2) + deltadiff, v13_mat, comb_indx, z, w)
                   - F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), x0(2), v1_mat, comb_indx, z, w) ) / deltadiff;
    deltax = - Hinv0 * Fdev0; //direction; DFP
    t = t0; // t0 will always be constant in every round of iteration
    x = x0 + t * deltax; //update x
    //v for each case based on x
    for(int k = 0; k < K; k++){
      v2_mat.col(k) = vForLA(v0, x(0), x(1), x(2), vectorise(Yrev.row(k), 0), tol_v);
    }//end first for loop
    //backtracking line search to determine stepsize
    Fx0 = F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), x0(2), v1_mat, comb_indx, z, w);
    Fx = F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), x(2), v2_mat, comb_indx, z, w);
    while(Fx > Fx0){
      t = t * tc;
      x = x0 + t * deltax; //update x
      //update v again
      for(int k = 0; k < K; k++){
        v2_mat.col(k) = vForLA(v0, x(0), x(1), x(2), vectorise(Yrev.row(k), 0), tol_v);
      }//end first for loop;
      Fx = F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), x(2), v2_mat, comb_indx, z, w);
    }//end while for backtracking line search
    for(int k = 0; k < K; k++){
      v21_mat.col(k) = vForLA(v0, x(0) + deltadiff, x(1), x(2), vectorise(Yrev.row(k), 0), tol_v);
      v22_mat.col(k) = vForLA(v0, x(0), x(1) + deltadiff, x(2), vectorise(Yrev.row(k), 0), tol_v);
      v23_mat.col(k) = vForLA(v0, x(0), x(1), x(2) + deltadiff, vectorise(Yrev.row(k), 0), tol_v);
    }//end first for loop
    Fdev(0) = ( F(Yrev, K, n, nobs_distinct_cases, x(0) + deltadiff, x(1), x(2), v21_mat, comb_indx, z, w)
                  - F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), x(2), v2_mat, comb_indx, z, w) ) / deltadiff;
    Fdev(1) = ( F(Yrev, K, n, nobs_distinct_cases, x(0), x(1) + deltadiff, x(2), v22_mat, comb_indx, z, w)
                  - F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), x(2), v2_mat, comb_indx, z, w) ) / deltadiff;
    Fdev(2) = ( F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), x(2) + deltadiff, v23_mat, comb_indx, z, w)
                  - F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), x(2), v2_mat, comb_indx, z, w) ) / deltadiff;
    s1 = x - x0;
    s2 = Fdev - Fdev0;
    Hinv = Hinv0 + s1 * s1.t() / as_scalar(s1.t() * s2 ) - (Hinv0 * s2 * s2.t() * Hinv0) /
      as_scalar(s2.t() * Hinv0 * s2); //DFP
    dis_x = as_scalar( sqrt( sum( pow(Fx - Fx0, 2) ) ) );
    //renew x0
    x0 = x;
    Hinv0 = Hinv;
    v1_mat = v2_mat;
    v11_mat = v21_mat;
    v12_mat = v22_mat;
    v13_mat = v23_mat;
    flag_step++;
  }//end while for Quasi-Newton
  return List::create(_["theta"] = x, _["H"] = Hinv);
}//end function LogitAGHMLE











