#include <algorithm>
#include <cassert>
#include <iostream>

#include "Eigen/Core"

#include "matrix.h"

using namespace std;

//////////////////////////////////////////////////
// NaiveMatrix                                  //
//////////////////////////////////////////////////

NaiveMatrix::NaiveMatrix(uint size)
    : BaseMatrix(size), m_(size, 1) {}

NaiveMatrix::NaiveMatrix(uint rows, uint cols)
   : BaseMatrix(rows, cols), m_(rows * cols, 0) {}

NaiveMatrix::NaiveMatrix(const NaiveMatrix& other)
    : BaseMatrix(other.rows_, other.cols_), m_(other.m_) {}

void NaiveMatrix::operator=(const NaiveMatrix& other) {
  rows_ = other.rows_;
  cols_ = other.cols_;
  m_ = other.m_;
}

float& NaiveMatrix::operator()(uint i, uint j) {
  assert(i < rows_);
  assert(j < cols_);
  return m_[cols_ * i + j];
}

float NaiveMatrix::operator()(uint i, uint j) const {
  assert(i < rows_);
  assert(j < cols_);
  return m_[cols_ * i + j];
}

float& NaiveMatrix::operator()(uint i) {
  assert(cols_ == 1);
  assert(i < rows_);
  return m_[i];
}

float NaiveMatrix::operator()(uint i) const {
  assert(cols_ == 1);
  assert(i < rows_);
  return m_[i];
}

NaiveMatrix NaiveMatrix::operator+(const NaiveMatrix& other) const {
  NaiveMatrix res(*this);
  res += other;
  return res;
}

void NaiveMatrix::operator+=(const NaiveMatrix& other) {
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  for(uint i = 0; i < rows_ * cols_; i++) {
    m_[i] += other.m_[i];
  }
}

NaiveMatrix NaiveMatrix::operator-(const NaiveMatrix& other) const {
  NaiveMatrix res(*this);
  res -= other;
  return res;
}

void NaiveMatrix::operator-=(const NaiveMatrix& other) {
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  for(uint i = 0; i < rows_ * cols_; i++) {
    m_[i] -= other.m_[i];
  }
}

NaiveMatrix NaiveMatrix::operator*(const NaiveMatrix& other) const {
  NaiveMatrix res(*this);
  res *= other;
  return res;
}

void NaiveMatrix::operator*=(const NaiveMatrix& other) {
  assert(cols_ == other.rows_);
  uint new_rows = rows_;
  uint new_cols = other.cols_;
  vector<float> new_m(new_rows * new_cols, 0.);
  for(uint i = 0; i < new_rows; i++) {
    for(uint j = 0; j < new_cols; j++) {
      for(uint k = 0; k < cols_; k++) {
        new_m[new_cols * i + j] += operator()(i, k) * other(k, j);
      }
    }
  }
  m_ = new_m;
  rows_ = new_rows;
  cols_ = new_cols;
}

NaiveMatrix NaiveMatrix::operator*(float c) const {
  NaiveMatrix res(*this);
  res *= c;
  return res;
}

void NaiveMatrix::operator*=(float c) {
  for(uint i = 0; i < m_.size(); i++) {
    m_[i] *= c;
  }
}

NaiveMatrix NaiveMatrix::CoeffWiseProduct(const NaiveMatrix& other) const {
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  NaiveMatrix res(*this);
  for(uint i = 0; i < res.m_.size(); i++) {
    res.m_[i] *= other.m_[i];
  }
  return res;
}

NaiveMatrix NaiveMatrix::Transpose() const {
  NaiveMatrix res(cols_, rows_);
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      res(j, i) = operator()(i, j);
    }
  }
  return res;
}

NaiveMatrix NaiveMatrix::ApplyFn(
    const pointer_to_unary_function<float, float>& fn) const {
  NaiveMatrix res(*this);
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      res(i, j) = fn(res(i, j));
    }
  }
  return res;
}

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

float& EigenMatrix::operator()(uint i, uint j) {
  assert(i < rows_);
  assert(j < cols_);
  return m_(i, j);
}

float EigenMatrix::operator()(uint i, uint j) const {
  assert(i < rows_);
  assert(j < cols_);
  return m_(i, j);
}

float& EigenMatrix::operator()(uint i) {
  assert(cols_ == 1);
  assert(i < rows_);
  return m_(i, 0);
}

float EigenMatrix::operator()(uint i) const {
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

EigenMatrix EigenMatrix::operator*(float c) const {
  EigenMatrix res(*this);
  res *= c;
  return res;
}

void EigenMatrix::operator*=(float c) {
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
    const pointer_to_unary_function<float, float>& fn) const {
  EigenMatrix res(*this);
  res.m_ = res.m_.unaryExpr(fn);
  return res;
}
