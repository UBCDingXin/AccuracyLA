//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp:plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
using namespace Rcpp;
using namespace arma;

/*

necessary functions for
 binary logit with random intercept

*/

// g(v) = -log(f(v)) for random intercept model
inline double gLogitInt(arma::vec y, double alpha, double v) {
  return y.n_elem * log(1 + exp(alpha + v)) - (alpha + v)*sum(y);
}
// first order derivative of g(v) wrt v
inline double gdev1LogitInt(arma::vec y, double alpha, double v) {
  return y.n_elem * exp(alpha+v) / (1+exp(alpha+v)) - sum(y);
}
// second order derivative of g(v) wrt v
inline double gdev2LogitInt(arma::vec y, double alpha, double v) {
  return y.n_elem * exp(alpha+v) / pow(1+exp(alpha+v), 2);
}

//F(alpha,sigma; v, Yrev) is negative loglik for Laplace approximation
double F(arma::mat Yrev, int K, int n, 
           arma::vec nobs_distinct_cases,
           double alpha, double sigma, arma::vec v_vec){
  vec tmp(K), y;
  double v, output;
  for (int k = 0; k < K; k++){
    v = v_vec(k);
    y = vectorise(Yrev.row(k), 0);
    double ratio = nobs_distinct_cases(k) / n;
    double expit_value2 = exp(alpha + v) / pow(1 + exp(alpha + v), 2);
    double tmp_value = pow(sigma, -2) + y.n_elem * expit_value2;
    tmp(k) = ratio * ( y.n_elem * log( 1 + exp(alpha + v) ) - ( alpha + v ) * sum(y) + 
              pow(v / sigma, 2) / 2 + 0.5 * log(pow(sigma, 2)) + 0.5 * log(tmp_value) );
  }
  output = sum(tmp);
  return output;
}//end function F
//first order partial derivative of F(alpha,sigma; v, y) wrt *alpha* and *sigma* respectively
arma::vec F1dev(arma::mat Yrev, int K, int n, 
                 arma::vec nobs_distinct_cases,
                 double alpha, double sigma, arma::vec v_vec){
  vec tmp1(K), tmp2(K), y;
  vec output(2);
  double v, vdev_alpha, vdev_sigma; //vdev_alpha and vdev_sigma arethe derivative of v wrt alpha and sigma
  for (int k = 0; k < K; k++){
    v = v_vec(k);
    y = vectorise(Yrev.row(k), 0);
    double ratio = nobs_distinct_cases(k) / n;
    double expit_value = exp(alpha + v) / ( 1 + exp(alpha + v) );
    double expit_value2 = exp(alpha + v) / pow(1 + exp(alpha + v), 2);
    double tmp_value1 = expit_value * ( 1 - exp(alpha + v) ) / pow(1 + exp(alpha + v), 2);
    double tmp_value2 = pow(sigma, -2) + y.n_elem * expit_value2;
    vdev_alpha = - expit_value2 / ( expit_value2 + pow(sigma, -2) );
    vdev_sigma = 2 * v / pow(sigma, 3) / 
      ( expit_value2 + pow(sigma, -2) );
    tmp1(k) = ratio * ( y.n_elem * expit_value * ( 1 + vdev_alpha ) - sum(y) * ( 1 + vdev_alpha ) + 
                v * pow(sigma, -2) * vdev_alpha + 
                0.5 * y.n_elem * tmp_value1 / tmp_value2 * ( 1 + vdev_alpha ) );
    tmp2(k) = ratio * ( ( y.n_elem * expit_value - sum(y) + pow(sigma, -2) ) * vdev_sigma -
                pow(sigma, -3) * pow(v, 2) + 1/sigma +
                0.5 * ( -2 * pow(sigma, -3) + y.n_elem * tmp_value1 * vdev_sigma ) / tmp_value2 );
  }//end for loop
  output(0) = sum(tmp1);//wrt alpha
  output(1) = sum(tmp2);//wrt sigma
  return output;
}


//root of eq.(8)
double vForLA(double v0, double alpha, double sigma, arma::vec y, double tol_v = 0.00001){
  //v0 is the initial value for random effect
  double dis_v = 999.0, v;
  while(dis_v > tol_v){
    v = v0 - ( gdev1LogitInt(y, alpha, v0) + v0 / pow(sigma, 2) ) /
      ( gdev2LogitInt(y, alpha, v0) + pow(sigma, -2) );//update v
    // dis_v = abs(v - v0);
    dis_v = abs(gdev1LogitInt(y, alpha, v) + v / pow(sigma, 2));
    v0 = v;
  }//end while
  return v;
}


//--------------------------------------------------------------------------
// [[Rcpp::export]]
List BiLogitLAMLE(arma::mat Y,double v0, arma::vec x0, double t0 = 1, double tc = 0.1, 
              double tol_v = 0.00001, double tol_x = 0.00001) {
  /*
   Y: response variable; a n by d matrix; 
    row -> different subjects; col -> multiple measurement for each subject;
   v0: initial values for random effect v
   x0: vector c(alpha, sigma), initial values for alpha and sigma
   t0: initial stepsize for backtracking line search
   tc: parameter for backtracking linear search; t = t0 * tc
   tol_v: convergence threshold for Newton-Raphson method
   tol_x: convergence threshold for Quasi Newton method
  */
  
  /*
  * first step: compute the number (K) of distinct cases of y (different sum of y); 
  * the K in formula (4) in the paper
  */
  int n = Y.n_rows, d = Y.n_cols; //dim of Y
  vec rowSumY_all = arma::sum(Y, 1); //summation of Y's rows
  vec rowSumY_uni = unique(rowSumY_all); //unique value of summation; 
  int K = rowSumY_uni.n_elem; //number of distinck cases
  vec nobs_distinct_cases(K,fill::zeros); //store the number of obs for each case
  mat Yrev(K, d, fill::zeros); //Yrev stores K distinct cases
  //determine row i belongs to which case
  for (int k = 0; k < K; k++){
    uvec tmp = find(rowSumY_all == rowSumY_uni(k));
    nobs_distinct_cases(k) = static_cast<double>(tmp.n_elem);
    Yrev.row(k) = Y.row(tmp(0));//for each case, only keep the first one
  }
  
  double dis_x = 999.0, t, Fx, Fx0;
  vec v1_vec(K), v2_vec(K); //v1 and v2 are random intercepts; for each row in Yrev, we will get a v.
  vec x;// x = (alpha,sigma) stores alpha (fixed intercept) and sigma;
  vec s1, s2, deltax, Fdev0, Fdev; // intermediate steps for Quasi Newton
  // mat H0 = eye<mat>(2,2), H;
  mat Hinv0 = eye<mat>(2,2), Hinv;
  int flag_step = 1;

  //v for each case based on x0
  for(int k = 0; k < K; k++){
    v1_vec(k) = vForLA(v0, x0(0), x0(1), vectorise(Yrev.row(k), 0), tol_v);
  }//end first for loop

  //Quasi Newton Method
  while (dis_x > tol_x){
    std::cout<<"Step:"<<flag_step<<"; dis:"<<dis_x<<std::endl;
    Fdev0 = F1dev(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), v1_vec);//derivative of F based on x0
    deltax = - Hinv0 * Fdev0; //direction; DFP
    t = t0; // t0 will always be constant in every round of iteration
    x = x0 + t * deltax; //update x
    //v for each case based on x
    for(int k = 0; k < K; k++){
      v2_vec(k) = vForLA(v0, x(0), x(1), vectorise(Yrev.row(k), 0), tol_v);
    }//end first for loop
    //backtracking line search to determine stepsize
    Fx0 = F(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), v1_vec);
    Fx = F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec);
    while(Fx > Fx0){
      t = t * tc;
      x = x0 + t * deltax; //update x
      //update v again
      for(int k = 0; k < K; k++){
        v2_vec(k) = vForLA(v0, x(0), x(1), vectorise(Yrev.row(k), 0), tol_v);
      }//end first for loop; 
      Fx = F(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec);
    }//end while for backtracking line search
    Fdev = F1dev(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec);//derivative of F based on x
    s1 = x - x0;
    s2 = Fdev - Fdev0;
    Hinv = Hinv0 + s1 * s1.t() / as_scalar(s1.t() * s2 ) - (Hinv0 * s2 * s2.t() * Hinv0) /
          as_scalar(s2.t() * Hinv0 * s2); //DFP
    // Hinv = Hinv0 + ( ( s1 - Hinv0 * s2 ) * s1.t() * Hinv0 )
    //           / as_scalar( s1.t() * Hinv0 * s2 ); //Broyden
    // dis_x = as_scalar( sqrt( sum( pow(Fdev, 2) ) ) );
    dis_x = as_scalar( sqrt( sum( pow(Fx - Fx0, 2) ) ) );
    //renew x0
    x0 = x;
    Hinv0 = Hinv;
    v1_vec = v2_vec;
    flag_step++;
  }//end while
  // std::cout<<Hinv0<<std::endl;
  return List::create(_["theta"] = x, _["H"] = Hinv);
}

// [[Rcpp::export]]
double FFun(arma::mat Y, double alpha, double sigma){
  //this function is for ploting negative loglik function
  int n = Y.n_rows, d = Y.n_cols; //dim of Y
  vec rowSumY_all = arma::sum(Y, 1); //summation of Y's rows
  vec rowSumY_uni = unique(rowSumY_all); //unique value of summation; 
  int K = rowSumY_uni.n_elem; //number of distinck cases
  vec nobs_distinct_cases(K,fill::zeros); //store the number of obs for each case
  mat Yrev(K, d, fill::zeros); //Yrev stores K distinct cases
  //determine row i belongs to which case
  for (int k = 0; k < K; k++){
    uvec tmp = find(rowSumY_all == rowSumY_uni(k));
    nobs_distinct_cases(k) = static_cast<double>(tmp.n_elem);
    Yrev.row(k) = Y.row(tmp(0));//for each case, only keep the first one
  }
  vec tmp(K);
  double v, output;
  for (int k = 0; k < K; k++){
    v = vForLA(0, alpha, sigma, vectorise(Yrev.row(k), 0), pow(10,-5));
    double ratio = nobs_distinct_cases(k) / n;
    double expit_value2 = exp(alpha + v) / pow(1 + exp(alpha + v), 2);
    double tmp_value = pow(sigma, -2) + d * expit_value2;
    tmp(k) = ratio * ( d * log( 1 + exp(alpha + v) ) - ( alpha + v ) * sum(Yrev.row(k)) + 
      pow(v / sigma, 2) / 2 + 0.5 * log(pow(sigma, 2)) + 0.5 * log(tmp_value) );
  }
  output = sum(tmp);
  return output;
}//end function F



//--------------------------------------------------------------------------
// AGH
//negative loglik for AGH
double FAGH(arma::mat Yrev, int K, int n, 
         arma::vec nobs_distinct_cases,
         double alpha, double sigma, arma::vec v_vec, 
         vec z, vec w){
  //z is a vector of nodes; w is a vector of weights
  int nq = z.n_elem; //number of nodes
  vec tmpG(nq), tmpApprox(K), y;
  double v, vforG, output, Rk, Gvalue, Wj;
  for (int k = 0; k < K; k++){
    v = v_vec(k);
    y = vectorise(Yrev.row(k), 0);
    double ratio = nobs_distinct_cases(k) / n;
    double expit_value2 = exp(alpha + v) / pow(1 + exp(alpha + v), 2);
    Rk = sqrt(y.n_elem * expit_value2 + pow(sigma,-2));
    for (int j = 0; j < nq; j++){
      Wj = exp(pow(z(j),2)) * w(j);
      vforG = v + z(j) / Rk;
      Gvalue = - gLogitInt(y, alpha, vforG) - pow(vforG, 2) / ( 2 * pow(sigma, 2) );
      tmpG(j) = exp(Gvalue) * Wj;
    }//end for loop
    tmpApprox(k) = ratio * ( log(sigma) + log(Rk) - log(sum(tmpG)) );
  }
  output = sum(tmpApprox);
  return output;
}//end function FAGH


//--------------------------------------------------------------------------
// [[Rcpp::export]]
List BiLogitAGHMLE(arma::mat Y, double v0, arma::vec x0, 
                      arma::vec z, arma::vec w, 
                      double t0 = 1, double tc = 0.1, double deltadiff = 1e-10,
                      double tol_v = 0.00001, double tol_x = 0.00001) {
  /*
   * first step: compute the number (K) of distinct cases of y (different sum of y); 
   * the K in formula (4) in the paper
   */
  int n = Y.n_rows, d = Y.n_cols; //dim of Y
  vec rowSumY_all = arma::sum(Y, 1); //summation of Y's rows
  vec rowSumY_uni = unique(rowSumY_all); //unique value of summation; 
  int K = rowSumY_uni.n_elem; //number of distinck cases
  vec nobs_distinct_cases(K,fill::zeros); //store the number of obs for each case
  mat Yrev(K, d, fill::zeros); //Yrev stores K distinct cases
  //determine row i belongs to which case
  for (int k = 0; k < K; k++){
    uvec tmp = find(rowSumY_all == rowSumY_uni(k));
    nobs_distinct_cases(k) = static_cast<double>(tmp.n_elem);
    Yrev.row(k) = Y.row(tmp(0));//for each case, only keep the first one
  }
  
  double dis_x = 999.0, t, Fx, Fx0;
  vec v1_vec(K), v11_vec(K), v12_vec(K), v2_vec(K), v21_vec(K), v22_vec(K); //v1 and v2 are random intercepts; for each row in Yrev, we will get a v.
  vec x;// x = (alpha,sigma) stores alpha (fixed intercept) and sigma;
  vec s1, s2, deltax(2), Fdev0(2), Fdev(2); // intermediate steps for Quasi Newton
  // mat H0 = eye<mat>(2,2), H;
  mat Hinv0 = eye<mat>(2,2), Hinv;
  int flag_step = 1;
  //v for each case based on x0
  for(int k = 0; k < K; k++){
    v1_vec(k) = vForLA(v0, x0(0), x0(1), vectorise(Yrev.row(k), 0), tol_v);
    v11_vec(k) = vForLA(v0, x0(0) + deltadiff, x0(1), vectorise(Yrev.row(k), 0), tol_v);
    v12_vec(k) = vForLA(v0, x0(0), x0(1) + deltadiff, vectorise(Yrev.row(k), 0), tol_v);
  }//end first for loop
  //Quasi Newton Method
  while (dis_x > tol_x){
    std::cout<<"Step:"<<flag_step<<"; dis:"<<dis_x<<std::endl;
    Fdev0(0) = ( FAGH(Yrev, K, n, nobs_distinct_cases, x0(0) + deltadiff, x0(1), v11_vec, z, w) 
                  - FAGH(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), v1_vec, z, w) ) / deltadiff;
    Fdev0(1) = ( FAGH(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1) + deltadiff, v12_vec, z, w) 
                   - FAGH(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), v1_vec, z, w) ) / deltadiff;
    deltax = - Hinv0 * Fdev0; //direction; DFP
    t = t0; // t0 will always be constant in every round of iteration
    x = x0 + t * deltax; //update x
    //v for each case based on x
    for(int k = 0; k < K; k++){
      v2_vec(k) = vForLA(v0, x(0), x(1), vectorise(Yrev.row(k), 0), tol_v);
    }//end first for loop
    //backtracking line search to determine stepsize
    Fx0 = FAGH(Yrev, K, n, nobs_distinct_cases, x0(0), x0(1), v1_vec, z, w);
    Fx = FAGH(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec, z, w);
    while(Fx > Fx0){
      t = t * tc;
      x = x0 + t * deltax; //update x
      //update v again
      for(int k = 0; k < K; k++){
        v2_vec(k) = vForLA(v0, x(0), x(1), vectorise(Yrev.row(k), 0), tol_v);
      }//end first for loop; 
      Fx = FAGH(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec, z, w);
    }//end while for backtracking line search
    //calculate numerical differentiation
    for(int k = 0; k < K; k++){
      v21_vec(k) = vForLA(v0, x(0) + deltadiff, x(1), vectorise(Yrev.row(k), 0), tol_v);
      v22_vec(k) = vForLA(v0, x(0), x(1) + deltadiff, vectorise(Yrev.row(k), 0), tol_v);
    }//end
    Fdev(0) = ( FAGH(Yrev, K, n, nobs_distinct_cases, x(0) + deltadiff, x(1), v21_vec, z, w) 
                   - FAGH(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec, z, w) ) / deltadiff;
    Fdev(1) = ( FAGH(Yrev, K, n, nobs_distinct_cases, x(0), x(1) + deltadiff, v22_vec, z, w) 
                   - FAGH(Yrev, K, n, nobs_distinct_cases, x(0), x(1), v2_vec, z, w) ) / deltadiff;
    
    s1 = x - x0;
    s2 = Fdev - Fdev0;
    Hinv = Hinv0 + s1 * s1.t() / as_scalar(s1.t() * s2 ) - (Hinv0 * s2 * s2.t() * Hinv0) /
      as_scalar(s2.t() * Hinv0 * s2); //DFP
    // dis_x = as_scalar( sqrt( sum( pow(Fdev, 2) ) ) );
    dis_x = as_scalar( sqrt( sum( pow(Fx - Fx0, 2) ) ) );
    //renew x0
    x0 = x;
    Hinv0 = Hinv;
    v1_vec = v2_vec;
    v11_vec = v21_vec;
    v12_vec = v22_vec;
    flag_step++;
  }//end while
  return List::create(_["theta"] = x, _["H"] = Hinv);
}//end AGH

