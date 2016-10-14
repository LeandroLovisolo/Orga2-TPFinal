#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <functional>
#include <initializer_list>
#include <random>
#include <vector>

#ifndef EMSCRIPTEN
#include "Eigen/Dense"
#include "gtest/gtest_prod.h"
#endif  // EMSCRIPTEN

//////////////////////////////////////////////////
// BaseMatrix                                   //
//////////////////////////////////////////////////

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
  BaseMatrix<Matrix>& Set(std::initializer_list<float> list);
  BaseMatrix<Matrix>& Zeros();
  BaseMatrix<Matrix>& Ones();
  BaseMatrix<Matrix>& Random();

  // Coefficient accessors
  virtual float& operator()(uint i, uint j) = 0;
  virtual float operator()(uint i, uint j) const = 0;
  virtual float& operator()(uint i) = 0;  // vector only
  virtual float operator()(uint i) const = 0;  // vector only

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
  virtual Matrix operator*(float c) const = 0;
  virtual void operator*=(float c) = 0;

  // Coefficient-wise product
  virtual Matrix CoeffWiseProduct(const Matrix& other) const = 0;

  // Transpose
  Matrix Transpose() const;

  // Coefficient-wise application of unary function
  Matrix ApplyFn(
      const std::pointer_to_unary_function<float, float>& fn) const;

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
    std::initializer_list<float> list) {
  assert(list.size() == rows_ * cols_);
  uint i = 0, j = 0;
  for(float x : list) {
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
  std::normal_distribution<float> distribution;
  for(uint i = 0; i < rows_; i++) {
    for(uint j = 0; j < cols_; j++) {
      operator()(i, j) = distribution(generator);
    }
  }
  return *this;
}

//////////////////////////////////////////////////
// NaiveMatrix                                  //
//////////////////////////////////////////////////

class NaiveMatrix : public BaseMatrix<NaiveMatrix> {
 public:
  NaiveMatrix(uint size);
  NaiveMatrix(uint rows, uint cols);
  NaiveMatrix(const NaiveMatrix& other);

  // Initializers
  void operator=(const NaiveMatrix& other);
  NaiveMatrix& Set(std::initializer_list<float> list) {
    BaseMatrix<NaiveMatrix>::Set(list);
    return *this;
  }
  NaiveMatrix& Zeros() { BaseMatrix<NaiveMatrix>::Zeros(); return *this; }
  NaiveMatrix& Ones() { BaseMatrix<NaiveMatrix>::Ones(); return *this; }
  NaiveMatrix& Random() { BaseMatrix<NaiveMatrix>::Random(); return *this; }

  // Coefficient accessors
  float& operator()(uint i, uint j);
  float operator()(uint i, uint j) const;
  float& operator()(uint i);  // vector only
  float operator()(uint i) const;  // vector only

  // Addition
  NaiveMatrix operator+(const NaiveMatrix& other) const;
  void operator+=(const NaiveMatrix& other);

  // Subtraction
  NaiveMatrix operator-(const NaiveMatrix& other) const;
  void operator-=(const NaiveMatrix& other);

  // Product
  NaiveMatrix operator*(const NaiveMatrix& other) const;
  void operator*=(const NaiveMatrix& other);

  // Scalar product
  NaiveMatrix operator*(float c) const;
  void operator*=(float c);

  // Coefficient-wise product
  NaiveMatrix CoeffWiseProduct(const NaiveMatrix& other) const;

  // Transpose
  NaiveMatrix Transpose() const;

  // Coefficient-wise application of unary function
  NaiveMatrix ApplyFn(
      const std::pointer_to_unary_function<float, float>& fn) const;

 private:
  std::vector<float> m_;
};

#ifndef EMSCRIPTEN

//////////////////////////////////////////////////
// SimdMatrix                                   //
//////////////////////////////////////////////////

class SimdMatrix : public BaseMatrix<SimdMatrix> {
 public:
  SimdMatrix(uint size);
  SimdMatrix(uint rows, uint cols);
  SimdMatrix(const SimdMatrix& other);

  // Initializers
  void operator=(const SimdMatrix& other);
  SimdMatrix& Set(std::initializer_list<float> list) {
    BaseMatrix<SimdMatrix>::Set(list);
    return *this;
  }
  SimdMatrix& Zeros() { BaseMatrix<SimdMatrix>::Zeros(); return *this; }
  SimdMatrix& Ones() { BaseMatrix<SimdMatrix>::Ones(); return *this; }
  SimdMatrix& Random() { BaseMatrix<SimdMatrix>::Random(); return *this; }

  // Coefficient accessors
  float& operator()(uint i, uint j);
  float operator()(uint i, uint j) const;
  float& operator()(uint i);  // vector only
  float operator()(uint i) const;  // vector only

  // Addition
  SimdMatrix operator+(const SimdMatrix& other) const;
  void operator+=(const SimdMatrix& other);

  // Subtraction
  SimdMatrix operator-(const SimdMatrix& other) const;
  void operator-=(const SimdMatrix& other);

  // Product
  SimdMatrix operator*(const SimdMatrix& other) const;
  void operator*=(const SimdMatrix& other);

  // Scalar product
  SimdMatrix operator*(float c) const;
  void operator*=(float c);

  // Coefficient-wise product
  SimdMatrix CoeffWiseProduct(const SimdMatrix& other) const;

  // Transpose
  SimdMatrix Transpose() const;

  // Coefficient-wise application of unary function
  SimdMatrix ApplyFn(
      const std::pointer_to_unary_function<float, float>& fn) const;

 private:
  std::vector<float> m_;
};

//////////////////////////////////////////////////
// EigenMatrix                                  //
//////////////////////////////////////////////////

class EigenMatrix : public BaseMatrix<EigenMatrix> {
 public:
  EigenMatrix(uint size);
  EigenMatrix(uint rows, uint cols);
  EigenMatrix(const EigenMatrix& other);

  // Initializers
  void operator=(const EigenMatrix& other);
  EigenMatrix& Set(std::initializer_list<float> list) {
    BaseMatrix<EigenMatrix>::Set(list);
    return *this;
  }
  EigenMatrix& Zeros() { BaseMatrix<EigenMatrix>::Zeros(); return *this; }
  EigenMatrix& Ones() { BaseMatrix<EigenMatrix>::Ones(); return *this; }
  EigenMatrix& Random() { BaseMatrix<EigenMatrix>::Random(); return *this; }

  // Coefficient accessors
  float& operator()(uint i, uint j);
  float operator()(uint i, uint j) const;
  float& operator()(uint i);  // vector only
  float operator()(uint i) const;  // vector only

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
  EigenMatrix operator*(float c) const;
  void operator*=(float c);

  // Coefficient-wise product
  EigenMatrix CoeffWiseProduct(const EigenMatrix& other) const;

  // Transpose
  EigenMatrix Transpose() const;

  // Coefficient-wise application of unary function
  EigenMatrix ApplyFn(
      const std::pointer_to_unary_function<float, float>& fn) const;

 private:
  Eigen::MatrixXf m_;
};

#endif  // EMSCRIPTEN

#endif  // __MATRIX_H__
