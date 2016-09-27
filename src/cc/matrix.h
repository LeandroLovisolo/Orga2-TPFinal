#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <functional>
#include <initializer_list>
#include <random>
#include <vector>

#include "Eigen/Dense"
#include "gtest/gtest_prod.h"

typedef unsigned int uint;

template <class Matrix>
class BaseMatrix {
 public:
  explicit BaseMatrix(uint size) : BaseMatrix(size, 1) {}  // vector only
  BaseMatrix(uint rows, uint cols) : rows_(rows), cols_(cols) {}

  inline uint rows() const { return rows_; }
  inline uint cols() const { return cols_; }
  inline uint size() const { assert(cols_ == 1); return rows_; }  // vector only

  bool operator==(const Matrix& other) const;
  bool operator!=(const Matrix& other) const;

  // Initializers
  virtual void operator=(const Matrix& other) = 0;
  BaseMatrix<Matrix>& Set(std::initializer_list<double> list);
  BaseMatrix<Matrix>& Zeros();
  BaseMatrix<Matrix>& Ones();
  BaseMatrix<Matrix>& Random();

  // Coefficient accessors
  virtual double& operator()(uint i, uint j) = 0;
  virtual double operator()(uint i, uint j) const = 0;
  virtual double& operator()(uint i) = 0;  // vector only
  virtual double operator()(uint i) const = 0;  // vector only

  // Addition
  virtual Matrix operator+(const Matrix& other) const = 0;
  virtual void operator+=(const Matrix& other) = 0;

  // Subtraction
  virtual Matrix operator-(const Matrix& other) const = 0;
  virtual void operator-=(const Matrix& other) = 0;

  // Product
  virtual Matrix operator*(const Matrix& other) const = 0;
  virtual void operator*=(const Matrix& other) = 0;

  // Scalar product
  virtual Matrix operator*(double c) const = 0;
  virtual void operator*=(double c) = 0;

  // Coefficient-wise product
  virtual Matrix CoeffWiseProduct(const Matrix& other) const = 0;

  // Transpose
  Matrix Transpose() const;

  // Coefficient-wise application of unary function
  Matrix ApplyFn(
      const std::pointer_to_unary_function<double, double>& fn) const;

 protected:
  uint rows_;
  uint cols_;
};

template <class Matrix>
bool BaseMatrix<Matrix>::operator==(const Matrix& other) const {
  if(rows_ != other.rows_ || cols_ != other.cols_) return false;
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      if(operator()(i, j) != other(i, j)) return false;
    }
  }
  return true;
}

template <class Matrix>
bool BaseMatrix<Matrix>::operator!=(const Matrix& other) const {
  return !(operator==(other));
}

template <class Matrix>
BaseMatrix<Matrix>& BaseMatrix<Matrix>::Set(
    std::initializer_list<double> list) {
  assert(list.size() == rows_ * cols_);
  uint i = 0, j = 0;
  for(double x : list) {
    operator()(i, j) = x;
    if(++j == cols_) {
      j = 0;
      i++;
    }
  }
  return *this;
}

template <class Matrix>
BaseMatrix<Matrix>& BaseMatrix<Matrix>::Zeros() {
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      this->operator()(i, j) = 0;
    }
  }
  return *this;
}

template <class Matrix>
BaseMatrix<Matrix>& BaseMatrix<Matrix>::Ones() {
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      this->operator()(i, j) = 1;
    }
  }
  return *this;
}

template <class Matrix>
BaseMatrix<Matrix>& BaseMatrix<Matrix>::Random() {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution;
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      operator()(i, j) = distribution(generator);
    }
  }
  return *this;
}

/*
class NaiveMatrix : public BaseMatrix {
 public:
  NaiveMatrix(uint rows, uint cols);
  double& operator()(uint i, uint j);
  // NaiveMatrix& operator+(const BaseMatrix& other);

 private:
  std::vector<double> m_;
};
*/

class EigenMatrix : public BaseMatrix<EigenMatrix> {
 public:
  EigenMatrix(uint size);
  EigenMatrix(uint rows, uint cols);
  EigenMatrix(const EigenMatrix& other);

  // Initializers
  void operator=(const EigenMatrix& other);
  EigenMatrix& Set(std::initializer_list<double> list) {
    BaseMatrix<EigenMatrix>::Set(list);
    return *this;
  }
  EigenMatrix& Zeros() { BaseMatrix<EigenMatrix>::Zeros(); return *this; }
  EigenMatrix& Ones() { BaseMatrix<EigenMatrix>::Ones(); return *this; }
  EigenMatrix& Random() { BaseMatrix<EigenMatrix>::Random(); return *this; }

  // Coefficient accessors
  double& operator()(uint i, uint j);
  double operator()(uint i, uint j) const;
  double& operator()(uint i);  // vector only
  double operator()(uint i) const;  // vector only

  // Addition
  EigenMatrix operator+(const EigenMatrix& other) const;
  void operator+=(const EigenMatrix& other);

  // Subtraction
  EigenMatrix operator-(const EigenMatrix& other) const;
  void operator-=(const EigenMatrix& other);

  // Product
  EigenMatrix operator*(const EigenMatrix& other) const;
  void operator*=(const EigenMatrix& other);

  // Scalar product
  EigenMatrix operator*(double c) const;
  void operator*=(double c);

  // Coefficient-wise product
  EigenMatrix CoeffWiseProduct(const EigenMatrix& other) const;

  // Transpose
  EigenMatrix Transpose() const;

  // Coefficient-wise application of unary function
  EigenMatrix ApplyFn(
      const std::pointer_to_unary_function<double, double>& fn) const;

 private:
  Eigen::MatrixXd m_;
};

void PrintTo(const EigenMatrix& m, ::std::ostream* os);

#endif // __MATRIX_H__
