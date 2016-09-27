#include <algorithm>
#include <cassert>
#include <iostream>

#include "Eigen/Core"

#include "matrix.h"

using namespace std;

//////////////////////////////////////////////////
// NaiveMatrix                                  //
//////////////////////////////////////////////////

/*
NaiveMatrix::NaiveMatrix(uint rows, uint cols)
   : BaseMatrix(rows, cols), m_(rows * cols, 0) {}

double& NaiveMatrix::operator()(uint i, uint j) {
  assert(i < rows_);
  assert(i < cols_);
  return m_[cols_ * j + i];
}

NaiveMatrix& NaiveMatrix::operator+(const BaseMatrix& other) {
  return NaiveMatrix(1, 1);
}
*/

//////////////////////////////////////////////////
// EigenMatrix                                  //
//////////////////////////////////////////////////

EigenMatrix::EigenMatrix(uint size)
    : BaseMatrix(size), m_(size, 1) {}

EigenMatrix::EigenMatrix(uint rows, uint cols)
    : BaseMatrix(rows, cols), m_(rows, cols) {}

EigenMatrix::EigenMatrix(const EigenMatrix& other)
    : BaseMatrix(other.rows_, other.cols_), m_(other.m_) {}

void EigenMatrix::operator=(const EigenMatrix& other) {
  rows_ = other.rows_;
  cols_ = other.cols_;
  m_ = other.m_;
}

double& EigenMatrix::operator()(uint i, uint j) {
  assert(i < rows_);
  assert(j < cols_);
  return m_(i, j);
}

double EigenMatrix::operator()(uint i, uint j) const {
  assert(i < rows_);
  assert(j < cols_);
  return m_(i, j);
}

double& EigenMatrix::operator()(uint i) {
  assert(cols_ == 1);
  assert(i < rows_);
  return m_(i, 0);
}

double EigenMatrix::operator()(uint i) const {
  assert(cols_ == 1);
  assert(i < rows_);
  return m_(i, 0);
}

EigenMatrix EigenMatrix::operator+(const EigenMatrix& other) const {
  EigenMatrix res(*this);
  res += other;
  return res;
}

void EigenMatrix::operator+=(const EigenMatrix& other) {
  m_ += other.m_;
}

EigenMatrix EigenMatrix::operator-(const EigenMatrix& other) const {
  EigenMatrix res(*this);
  res -= other;
  return res;
}

void EigenMatrix::operator-=(const EigenMatrix& other) {
  m_ -= other.m_;
}

EigenMatrix EigenMatrix::operator*(const EigenMatrix& other) const {
  EigenMatrix res(*this);
  res *= other;
  return res;
}

void EigenMatrix::operator*=(const EigenMatrix& other) {
  m_ *= other.m_;
  rows_ = m_.rows();
  cols_ = m_.cols();
}

EigenMatrix EigenMatrix::operator*(double c) const {
  EigenMatrix res(*this);
  res *= c;
  return res;
}

void EigenMatrix::operator*=(double c) {
  m_ *= c;
}

EigenMatrix EigenMatrix::CoeffWiseProduct(const EigenMatrix& other) const {
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  EigenMatrix res(*this);
  res.m_ = res.m_.cwiseProduct(other.m_);
  return res;
}

EigenMatrix EigenMatrix::Transpose() const {
  EigenMatrix res(*this);
  res.m_.transposeInPlace();
  res.rows_ = cols_;
  res.cols_ = rows_;
  return res;
}

EigenMatrix EigenMatrix::ApplyFn(
    const pointer_to_unary_function<double, double>& fn) const {
  EigenMatrix res(*this);
  res.m_ = res.m_.unaryExpr(fn);
  return res;
}

void PrintTo(const EigenMatrix& m, ::std::ostream* os) {
  string s = "\n";
  for(uint i = 0; i < m.rows(); i++) {
    if(i > 0) s += "\n";
    for(uint j = 0; j < m.cols(); j++) {
      if(j > 0) s += " ";
      s += to_string(m(i, j));
    }
  }
  *os << s;
}
