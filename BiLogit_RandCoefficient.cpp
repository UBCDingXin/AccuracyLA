//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp:plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
using namespace Rcpp;
using namespace arma;

//function eta; part of y's pdf
inline double eta(arma::vec x, arma::vec beta, arma::vec v){
  // x is a c by 1 vector corresponding to the j-th observation for i-th cluster
  vec tmp(v.n_elem,fill::ones);
  tmp.subvec(1,v.n_elem-1) = x;
  return as_scalar( ( v + beta ).t() * tmp );
  // vec tmp = beta.subvec(1, beta.n_elem-1) + v.subvec(1, v.n_elem-1);
  // return v(0) + beta(0) + as_scalar(tmp.t() * x);
}//end function eta


// g(v) = -log(f(v))
double g(arma::vec y, arma::mat X, arma::vec beta, arma::vec v){
  //y is d * 1; X is d * c;
  int d = y.n_elem;
  vec tmp(d);
  double tmpeta;
  for (int j = 0; j < d; j++){
    tmpeta = eta( vectorise(X.row(j), 0), beta, v );
    tmp(j) = log( 1 + exp( y(j) *  tmpeta ) ) - y(j) * tmpeta;
  }//
  return sum(tmp);
}//end of g()
//first derivative of g(v) wrt v
arma::vec gdev1(arma::vec y, arma::mat X, arma::vec beta, arma::vec v) {
  //y is d by 1; X is d by c
  int d = y.n_elem;
  int c = v.n_elem - 1;//r = c + 1
  double valueExp, tmpeta;
  vec output(c + 1, fill::zeros), etadev1( c + 1, fill::ones);
  // etadev1(0) = 1;
  for (int j = 0; j < d; j++){
    tmpeta = eta( vectorise(X.row(j), 0), beta, v );//evl eta function
    valueExp = ( y(j) * exp( y(j) * tmpeta ) ) / ( 1 + exp( y(j) * tmpeta ) ) - y(j);
    etadev1.subvec(1, c) = vectorise(X.row(j), 0);
    output = output + valueExp * etadev1;
  }//
  return output;//return a c+1 by 1 vector
}//end of gdev1
//second derivative of g(v) wrt v
arma::mat gdev2(arma::vec y, arma::mat X, arma::vec beta, arma::vec v) {
  int d = y.n_elem;
  int c = v.n_elem - 1;//r = c + 1
  double valueExp, tmpeta;
  mat output(c + 1, c + 1, fill::zeros);
  vec etadev1( c + 1 );
  etadev1(0) = 1;
  for (int j = 0; j < d; j++){
    tmpeta = eta( vectorise(X.row(j), 0), beta, v );//evl eta function
    // valueExp = ( pow(y(j), 3) * exp( y(j) * tmpeta ) * ( 1 - exp( y(j) * tmpeta ) ) ) 
    //   / pow( 1 + exp( y(j) * tmpeta ), 3 );//wrong formula
    valueExp = ( pow(y(j), 2) * exp( y(j) * tmpeta ) )  
      / pow( 1 + exp( y(j) * tmpeta ), 2 );
    etadev1.subvec(1, c) = vectorise(X.row(j), 0);
    output = output + valueExp * etadev1 * etadev1.t();
  }//
  return output;//return a c+1 by c+1 matrix
}//end of gdev2

//root of eq.(8); compute \tilde{v}
arma::vec vForLA(arma::vec v0, arma::vec beta, arma::vec sigma, 
                 arma::vec y, arma::mat X,
                 double tol_v = 0.00001){
  //v0 is the initial value for random effect
  int r = v0.n_elem;
  // int c = r - 1;
  mat Sigma(r,r,fill::zeros);
  Sigma.diag() = pow(sigma, 2);
  int flag = 1;
  double dis_v = 999.0;
  vec v = v0;
  while((dis_v > tol_v) && (flag < 10000)){
  // while((dis_v > tol_v)){
    // std::cout<<flag<<";"<<dis_v<<std::endl;
    v = v0 - ( gdev2(y, X, beta, v0) + Sigma.i() ).i() *
      ( gdev1(y, X, beta, v0) + Sigma.i() * v0 );//update v
    // dis_v = abs(v - v0);
    dis_v = as_scalar(sum(abs(gdev1(y, X, beta, v) + Sigma.i() * v)));
    v0 = v;
    flag++;
  }//end while
  return v;
}//end vForLA


//Probability limit of negative log-likelihood function
double F(arma::mat Y, arma::cube XCube, int K, 
         arma::vec beta, arma::vec sigma, arma::mat v_mat,
         arma::mat comb_indx, vec z, vec w){
  // each obs is a distinct case;
  // r = c + 1
  // Y is K * d matrix; XCube is a d * c * K cube
  // v_mat should be a r * K matrix; comb_indx is a r * nq^r matrix
  // z is a vector of nq nodes; w is a vector of weights
  int nq = z.n_elem; //number of nodes
  int r = v_mat.n_rows; // number of random effects
  int d = Y.n_cols; // cluster size d;
  vec tmpG(pow(nq, r)), tmpApprox(K), y(d);
  double output, Gvalue, Wj;
  vec v, vforG, zj(r), wj(r);
  mat Rk, Sigma(r, r,fill::zeros), X(d, r - 1);
  Sigma.diag() = pow(sigma, 2);
  for (int k = 0; k < K; k++){
    v = v_mat.col(k);
    y = vectorise(Y.row(k), 0);
    X = XCube.slice(k);
    double ratio = static_cast<double>(1) / K;
    Rk = chol(gdev2(y, X, beta, v) + Sigma.i());
    for (int j = 0; j < pow(nq, r); j++){
      for (int jj = 0; jj < r; jj++){
        zj(jj) = z(comb_indx(jj, j)); //get the nodes combination
        wj(jj) = w(comb_indx(jj, j)); //get the weights combination
      }//end for loop: jj from 0 to r
      Wj = as_scalar(exp(sum(pow(zj, 2))) * prod(wj));
      vforG = v + Rk.i() * zj;
      Gvalue = - g(y, X, beta, vforG) - 0.5 * as_scalar(vforG.t() * Sigma.i() * vforG);
      tmpG(j) = exp(Gvalue) * Wj;
    }//end for loop: j from 0 to nq^r
    tmpApprox(k) = ratio * ( 0.5 * log(det(Sigma)) + log(det(Rk)) - log(sum(tmpG)) );
  }//end for loop: k from 0 to K
  output = sum(tmpApprox);
  return output;
}//end F


//--------------------------------------------------------------------------
//Quasi-Newton method
// [[Rcpp::export]]
List LogitAGHMLE(arma::mat Y, arma::cube XCube, double v0_num, arma::vec x0, 
                 arma::vec z, arma::vec w, arma::mat comb_indx,
                 double t0 = 1, double tc = 0.1, double deltadiff = 1e-10,
                 double tol_v = 0.00001, double tol_x = 0.00001) {
  // Y is K * d matrix; XCube is a d * c * K cube
  //x0 is (beta,sigma) a 2r by 1 vector
  // comb_indx is a r * nq^r matrix
  // z is a vector of nq nodes; w is a vector of weights
  int K = Y.n_rows, d = Y.n_cols; //dim of Y; K * d
  int r = x0.n_elem / 2;
  // int c = r - 1;
  vec v0(r, fill::ones);
  v0 = v0 * v0_num;  
  double dis_x = 999.0, t, Fx, Fx0;
  mat v1_mat(r, K), v2_mat(r, K); //random effect
  cube v11_cube(r, K, 2 * r), v21_cube(r, K, 2 * r);//random effect for differentiation
  vec x;// x = (beta,sigma)
  vec s1, s2, deltax(2 * r), Fdev0(2 * r), Fdev(2 * r), deltaxFordiff; // intermediate steps for Quasi Newton
  mat Hinv0 = eye<mat>(2 * r, 2 * r), Hinv;
  int flag_step = 1;
  //v for each case based on x0
  for(int k = 0; k < K; k++){
    v1_mat.col(k) = vForLA(v0, x0.subvec(0, r - 1), x0.subvec(r, 2 * r - 1 ),
               vectorise(Y.row(k), 0), XCube.slice(k), tol_v);
    for (int h = 0; h < 2 * r; h++){
      deltaxFordiff = x0;
      deltaxFordiff(h) = deltaxFordiff(h) + deltadiff;
      v11_cube(span(0,r-1), span(k,k), span(h,h)) = 
        vForLA(v0, deltaxFordiff.subvec(0,r-1), deltaxFordiff.subvec(r, 2 * r - 1 ),
               vectorise(Y.row(k), 0), XCube.slice(k), tol_v);
    }//end for loop: a slight change added to each variable independently
  }//end loop: prepare for numerical differentiation of F at x0

  //Quasi Newton Method
  while (dis_x > tol_x){
    // std::cout<<"Step:"<<flag_step<<"; dis:"<<dis_x<<std::endl;
    for (int h = 0; h < 2 * r; h++){
      deltaxFordiff = x0;
      deltaxFordiff(h) = deltaxFordiff(h) + deltadiff;
      Fdev0(h) = ( F(Y, XCube, K, deltaxFordiff.subvec(0,r-1), deltaxFordiff.subvec(r, 2 * r - 1 ), v11_cube.slice(h), comb_indx, z, w)
                    - F(Y, XCube, K, x0.subvec(0,r-1), x0.subvec(r, 2 * r - 1 ), v1_mat, comb_indx, z, w) ) /
                      deltadiff;
    }//end for loop: Numerical differentiation at x0
    deltax = - Hinv0 * Fdev0; //direction; DFP
    // std::cout<<Hinv0<<std::endl;
    t = t0; // t0 will always be constant in every round of iteration
    x = x0 + t * deltax; //update x
    //v for each case based on first updated x;
    for(int k = 0; k < K; k++){
      v2_mat.col(k) = vForLA(v0, x.subvec(0, r - 1), x.subvec(r, 2 * r - 1 ),
                 vectorise(Y.row(k), 0), XCube.slice(k), tol_v);
    }//end for loop to calculate v2 based on first updated x;
    //backtracking line search to determine stepsize
    Fx0 = F(Y, XCube, K, x0.subvec(0,r-1), x0.subvec(r, 2 * r - 1 ), v1_mat, comb_indx, z, w);
    Fx = F(Y, XCube, K, x.subvec(0,r-1), x.subvec(r, 2 * r - 1 ), v2_mat, comb_indx, z, w);
    while(Fx > Fx0){
      t = t * tc;
      x = x0 + t * deltax; //update x
      //update v again
      for(int k = 0; k < K; k++){
        v2_mat.col(k) = vForLA(v0, x.subvec(0, r - 1), x.subvec(r, 2 * r - 1 ),
                   vectorise(Y.row(k), 0), XCube.slice(k), tol_v);
      }//end first for loop;
      Fx = F(Y, XCube, K, x.subvec(0,r-1), x.subvec(r, 2 * r - 1 ), v2_mat, comb_indx, z, w);
    }//end while for backtracking line search
    for(int k = 0; k < K; k++){
      for (int h = 0; h < 2 * r; h++){
        deltaxFordiff = x;
        deltaxFordiff(h) = deltaxFordiff(h) + deltadiff;
        v21_cube(span(0,r-1), span(k,k), span(h,h)) =
          vForLA(v0, deltaxFordiff.subvec(0,r-1), deltaxFordiff.subvec(r, 2 * r - 1 ),
                 vectorise(Y.row(k), 0), XCube.slice(k), tol_v);
      }//end for loop: a slight change added to each variable independently
    }//end for loop: prepare for numerical differentiation at x
    for (int h = 0; h < 2 * r; h++){
      deltaxFordiff = x;
      deltaxFordiff(h) = deltaxFordiff(h) + deltadiff;
      Fdev(h) = ( F(Y, XCube, K, deltaxFordiff.subvec(0,r-1), deltaxFordiff.subvec(r, 2 * r - 1 ), v21_cube.slice(h), comb_indx, z, w)
                     - F(Y, XCube, K, x.subvec(0,r-1), x.subvec(r, 2 * r - 1 ), v2_mat, comb_indx, z, w) ) /
                       deltadiff;
    }//end for loop: Numerical differentiation at x
    s1 = x - x0;
    s2 = Fdev - Fdev0;
    std::cout<<as_scalar(s1.t() * s2 )<<std::endl;
    Hinv = Hinv0 + s1 * s1.t() / as_scalar(s1.t() * s2 ) - (Hinv0 * s2 * s2.t() * Hinv0) /
      as_scalar(s2.t() * Hinv0 * s2); //DFP
    // Hinv = Hinv0 + ( ( s1 - Hinv0 * s2 ) * s1.t() * Hinv0 )
    //           / as_scalar( s1.t() * Hinv0 * s2 ); //Broyden
    dis_x = as_scalar( sqrt( sum( pow(Fx - Fx0, 2) ) ) );
    //renew x0
    x0 = x;
    Hinv0 = Hinv;
    v1_mat = v2_mat;
    v11_cube = v21_cube;
    flag_step++;
  }//end Quasi-Newton method
  return List::create(_["theta"] = x, _["Hinv"] = Hinv);
}//end Quasi-Newton method








