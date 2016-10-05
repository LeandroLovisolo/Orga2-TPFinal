#include <memory>

#include "ini.h"
#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "matrix.h"
#include "simd_matrix.h"

using namespace std;

//////////////////////////////////////////////////
// MatrixTest                                   //
//////////////////////////////////////////////////

template <typename Matrix>
class MatrixTest : public ::testing::Test {
 public:
  MatrixTest() {
    static_assert(std::is_base_of<BaseMatrix<Matrix>, Matrix>::value,
                  "Matrix not derived from BaseMatrix");
  }

  Matrix CreateMatrix(uint rows, uint cols) {
    return Matrix(rows, cols);
  }

  Matrix CreateVector(uint size) {
    return Matrix(size);
  }
};

typedef ::testing::Types<NaiveMatrix, SimdMatrix, EigenMatrix> MatrixTypes;

TYPED_TEST_CASE(MatrixTest, MatrixTypes);

#define Matrix this->CreateMatrix
#define Vector this->CreateVector

TYPED_TEST(MatrixTest, Rows) {
  EXPECT_EQ(2, Matrix(2, 3).rows());
}

TYPED_TEST(MatrixTest, Cols) {
  EXPECT_EQ(3, Matrix(2, 3).cols());
}

TYPED_TEST(MatrixTest, Size) {
  EXPECT_EQ(4, Vector(4).size());
}

TYPED_TEST(MatrixTest, Accessor) {
  auto m = Matrix(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  EXPECT_EQ(1, m(0, 0));
  EXPECT_EQ(2, m(0, 1));
  EXPECT_EQ(3, m(1, 0));
  EXPECT_EQ(4, m(1, 1));
}

TYPED_TEST(MatrixTest, AccessorVector) {
  auto m = Vector(3);
  m(0) = 1;
  m(1) = 2;
  m(2) = 3;
  EXPECT_EQ(1, m(0));
  EXPECT_EQ(2, m(1));
  EXPECT_EQ(3, m(2));
}

TYPED_TEST(MatrixTest, Set) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  EXPECT_EQ(1, m(0, 0));
  EXPECT_EQ(2, m(0, 1));
  EXPECT_EQ(3, m(1, 0));
  EXPECT_EQ(4, m(1, 1));
}

TYPED_TEST(MatrixTest, Comparison) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  EXPECT_NE(m, Matrix(2, 3));
  EXPECT_NE(m, Matrix(3, 2));
  EXPECT_NE(m, Matrix(3, 3));
  EXPECT_NE(m, Matrix(2, 2).Set({ 1, 1, 1, 1 }));
  EXPECT_EQ(m, Matrix(2, 2).Set({ 1, 2, 3, 4 }));
}

TYPED_TEST(MatrixTest, SetZeros) {
  EXPECT_EQ(Matrix(2, 2).Zeros(),
            Matrix(2, 2).Set({ 0, 0, 0, 0 }));
}

TYPED_TEST(MatrixTest, SetOnes) {
  EXPECT_EQ(Matrix(2, 2).Ones(),
            Matrix(2, 2).Set({ 1, 1, 1, 1 }));
}

TYPED_TEST(MatrixTest, Assignment) {
  auto m = Matrix(2, 2);
  auto n = Matrix(3, 4).Set({ 1, 2,  3,  4,
                              5, 6,  7,  8,
                              9, 10, 11, 12 });
  m = n;
  EXPECT_EQ(m, n);
}

TYPED_TEST(MatrixTest, Addition) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  auto n = Matrix(2, 2).Set({ 5, 6, 7, 8 });
  auto o = Matrix(2, 2).Set({ 6, 8, 10, 12 });
  EXPECT_EQ(m + n, o);
  m += n;
  EXPECT_EQ(m, o);
}

TYPED_TEST(MatrixTest, Subtraction) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  auto n = Matrix(2, 2).Set({ 8, 7, 6, 5 });
  auto o = Matrix(2, 2).Set({ -7, -5, -3, -1 });
  EXPECT_EQ(m - n, o);
  m -= n;
  EXPECT_EQ(m, o);
}

TYPED_TEST(MatrixTest, Product) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  auto n = Matrix(2, 2).Set({ 8, 7, 6, 5 });
  auto o = Matrix(2, 2).Set({ 20, 17, 48, 41 });
  EXPECT_EQ(m * n, o);
  m *= n;
  EXPECT_EQ(m, o);
}

TYPED_TEST(MatrixTest, ProductDifferentDimensions) {
  auto m = Matrix(4, 3).Set({ 1, 2, 3,
                              4, 5, 6,
                              7, 8, 9,
                              10, 11, 12 });
  auto n = Matrix(3, 2).Set({ 6, 5,
                              4, 3,
                              2, 1 });
  auto o = Matrix(4, 2).Set({ 20, 14,
                              56, 41,
                              92, 68,
                              128, 95 });
  EXPECT_EQ(m * n, o);
  m *= n;
  EXPECT_EQ(m, o);
}

TYPED_TEST(MatrixTest, ScalarProduct) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  auto n = Matrix(2, 2).Set({ 2, 4, 6, 8 });
  EXPECT_EQ(m * 2, n);
  m *= 2;
  EXPECT_EQ(m, n);
}

TYPED_TEST(MatrixTest, CoefficientWiseProduct) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  auto n = Matrix(2, 2).Set({ 2, 3, 4, 5 });
  auto o = Matrix(2, 2).Set({ 2, 6, 12, 20 });
  EXPECT_EQ(m.CoeffWiseProduct(n), o);
}

TYPED_TEST(MatrixTest, Transpose) {
  auto m = Matrix(3, 2).Set({ 1, 2,
                              3, 4,
                              5, 6 });
  auto n = Matrix(2, 3).Set({ 1, 3, 5,
                              2, 4, 6 });
  EXPECT_EQ(m.Transpose(), n);
}

float square(float x) { return x * x; }

TYPED_TEST(MatrixTest, ApplyFn) {
  auto m = Matrix(2, 2).Set({ 1, 2, 3, 4 });
  auto n = Matrix(2, 2).Set({ 1, 4, 9, 16 });
  EXPECT_EQ(m.ApplyFn(ptr_fun(square)), n);
}

template <typename M>
void PrintMatrix(const BaseMatrix<M>& m, ostream* os) {
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

void PrintTo(const NaiveMatrix& m, ::std::ostream* os) {
  PrintMatrix<NaiveMatrix>(m, os);
}

void PrintTo(const EigenMatrix& m, ::std::ostream* os) {
  PrintMatrix<EigenMatrix>(m, os);
}

//////////////////////////////////////////////////
// SimdMatrixTest                               //
//////////////////////////////////////////////////

class SimdMatrixTest : public ::testing::Test {
 protected:
  void TestScalarProduct(uint size) {
    vector<float> m = IncreasingVector(1, size);
    float c = 3;
    vector<float> s = ScalarProduct(m, c);
    simd_matrix_scalar_product(size, m.data(), c);
    EXPECT_EQ(m, s);
  }

  void TestAddition(uint size) {
    vector<float> m = IncreasingVector(1, size);
    vector<float> n = IncreasingVector(size + 1, size);
    vector<float> s = Sum(m, n);
    simd_matrix_addition(size, m.data(), n.data());
    EXPECT_EQ(m, s);
  }

  void TestSubtraction(uint size) {
    vector<float> m = IncreasingVector(1, size);
    vector<float> n = IncreasingVector(size + 1, size);
    vector<float> s = Subtract(m, n);
    simd_matrix_subtraction(size, m.data(), n.data());
    EXPECT_EQ(m, s);
  }

  void TestCoeffWiseProduct(uint size) {
    vector<float> m = IncreasingVector(1, size);
    vector<float> n = IncreasingVector(size + 1, size);
    vector<float> p = CoeffWiseProduct(m, n);
    simd_matrix_coeff_wise_product(size, m.data(), n.data());
    EXPECT_EQ(m, p);
  }

  void TestTranspose(uint rows, uint cols) {
    vector<float> m = IncreasingVector(1, rows * cols);
    vector<float> mt(rows * cols, 0);
    vector<float> expected_mt = Transpose(rows, cols, m);
    simd_matrix_transpose(rows, cols, m.data(), mt.data());
    EXPECT_EQ(mt, expected_mt);
  }

  void TestProduct(uint rows, uint cols1, uint cols2) {
    vector<float> m = IncreasingVector(1, rows * cols1);
    vector<float> n = IncreasingVector(cols1 * cols2 + 1, cols1 * cols2);
    vector<float> nt = Transpose(cols1, cols2, n);
    vector<float> mn(rows * cols2, 0);
    vector<float> expected_mn = Product(rows, cols1, cols2, m, n);
    simd_matrix_product(rows, cols1, cols2, m.data(), nt.data(), mn.data());
    EXPECT_EQ(mn, expected_mn);
  }

 private:
  vector<float> IncreasingVector(float start, uint size) {
    vector<float> v;
    while(size > 0) {
      v.push_back(start);
      start += 1;
      size--;
    }
    return v;
  }

  vector<float> ScalarProduct(const vector<float> &v, float c) {
    vector<float> s(v);
    for(float &f : s) {
      f *= c;
    }
    return s;
  }

  vector<float> Sum(const vector<float> &v, const vector<float>& w) {
    assert(v.size() == w.size());
    vector<float> s;
    for(int i = 0; i < v.size(); i++) s.push_back(v[i] + w[i]);
    return s;
  }

  vector<float> Subtract(const vector<float>& v, const vector<float>& w) {
    return Sum(v, ScalarProduct(w, -1));
  }

  vector<float> CoeffWiseProduct(
      const vector<float> &v, const vector<float>& w) {
    assert(v.size() == w.size());
    vector<float> x;
    for(int i = 0; i < v.size(); i++) {
      x.push_back(v[i] * w[i]);
    }
    return x;
  }

  vector<float> Transpose(uint rows, uint cols, const vector<float>& v) {
    assert(rows * cols == v.size());
    vector<float> w(v.size());
    for(uint i = 0; i < rows; i++) {
      for(uint j = 0; j < cols; j++) {
        w[j * rows + i] = v[i * cols + j];
      }
    }
    return w;
  }

  vector<float> Product(uint rows, uint cols1, uint cols2,
                        const vector<float>& m, const vector<float>& n) {
    vector<float> mn(rows * cols2, 0);
    for(uint i = 0; i < rows; i++) {
      for(uint j = 0; j < cols2; j++) {
        for(uint k = 0; k < cols1; k++) {
          mn[cols2 * i + j] += m[i * cols1 + k] * n[k * cols2 + j];
        }
      }
    }
    return mn;
  }

  FRIEND_TEST(SimdMatrixTest, Transpose);
  FRIEND_TEST(SimdMatrixTest, Product);
};

// Scalar product: minimum length
TEST_F(SimdMatrixTest, ScalarProductLength4) { TestScalarProduct(4); }

// Scalar product: lenghts divisible by 4
TEST_F(SimdMatrixTest, ScalarProductLength8)  { TestScalarProduct(8); }
TEST_F(SimdMatrixTest, ScalarProductLength12) { TestScalarProduct(12); }

// Scalar product: lenghts non-divisible by 4
TEST_F(SimdMatrixTest, ScalarProductLength5)  { TestScalarProduct(5); }
TEST_F(SimdMatrixTest, ScalarProductLength6)  { TestScalarProduct(6); }
TEST_F(SimdMatrixTest, ScalarProductLength7)  { TestScalarProduct(7); }
TEST_F(SimdMatrixTest, ScalarProductLength9)  { TestScalarProduct(9); }
TEST_F(SimdMatrixTest, ScalarProductLength10) { TestScalarProduct(10); }
TEST_F(SimdMatrixTest, ScalarProductLength11) { TestScalarProduct(11); }

// Addition: minimum length
TEST_F(SimdMatrixTest, AdditionLength4) { TestAddition(4); }

// Addition: lenghts divisible by 4
TEST_F(SimdMatrixTest, AdditionLength8)  { TestAddition(8); }
TEST_F(SimdMatrixTest, AdditionLength12) { TestAddition(12); }

// Addition: lenghts non-divisible by 4
TEST_F(SimdMatrixTest, AdditionLength5)  { TestAddition(5); }
TEST_F(SimdMatrixTest, AdditionLength6)  { TestAddition(6); }
TEST_F(SimdMatrixTest, AdditionLength7)  { TestAddition(7); }
TEST_F(SimdMatrixTest, AdditionLength9)  { TestAddition(9); }
TEST_F(SimdMatrixTest, AdditionLength10) { TestAddition(10); }
TEST_F(SimdMatrixTest, AdditionLength11) { TestAddition(11); }

// Subtraction: minimum length
TEST_F(SimdMatrixTest, SubtractionLength4) { TestSubtraction(4); }

// Subtraction: lenghts divisible by 4
TEST_F(SimdMatrixTest, SubtractionLength8)  { TestSubtraction(8); }
TEST_F(SimdMatrixTest, SubtractionLength12) { TestSubtraction(12); }

// Subtraction: lenghts non-divisible by 4
TEST_F(SimdMatrixTest, SubtractionLength5)  { TestSubtraction(5); }
TEST_F(SimdMatrixTest, SubtractionLength6)  { TestSubtraction(6); }
TEST_F(SimdMatrixTest, SubtractionLength7)  { TestSubtraction(7); }
TEST_F(SimdMatrixTest, SubtractionLength9)  { TestSubtraction(9); }
TEST_F(SimdMatrixTest, SubtractionLength10) { TestSubtraction(10); }
TEST_F(SimdMatrixTest, SubtractionLength11) { TestSubtraction(11); }

// Coefficient wise product: minimum length
TEST_F(SimdMatrixTest, CoeffWiseProductLength4) { TestCoeffWiseProduct(4); }

// Coefficient wise product: lenghts divisible by 4
TEST_F(SimdMatrixTest, CoeffWiseProductLength8)  { TestCoeffWiseProduct(8); }
TEST_F(SimdMatrixTest, CoeffWiseProductLength12) { TestCoeffWiseProduct(12); }

// Coefficient wise product: lenghts non-divisible by 4
TEST_F(SimdMatrixTest, CoeffWiseProductLength5)  { TestCoeffWiseProduct(5); }
TEST_F(SimdMatrixTest, CoeffWiseProductLength6)  { TestCoeffWiseProduct(6); }
TEST_F(SimdMatrixTest, CoeffWiseProductLength7)  { TestCoeffWiseProduct(7); }
TEST_F(SimdMatrixTest, CoeffWiseProductLength9)  { TestCoeffWiseProduct(9); }
TEST_F(SimdMatrixTest, CoeffWiseProductLength10) { TestCoeffWiseProduct(10); }
TEST_F(SimdMatrixTest, CoeffWiseProductLength11) { TestCoeffWiseProduct(11); }

// Test method SimdMatrixTest::Transpose() used to generate test cases
TEST_F(SimdMatrixTest, Transpose) {
  uint rows = 2, cols = 3;
  vector<float> m  = { 1, 2, 3,
                       4, 5, 6 };
  vector<float> mt = { 1, 4,
                       2, 5,
                       3, 6 };
  EXPECT_EQ(mt, Transpose(rows, cols, m));
}

// Transpose: minimum rows/cols
TEST_F(SimdMatrixTest, Transpose4by1) { TestTranspose(4, 1); }

// Transpose: rows divisible by 4
TEST_F(SimdMatrixTest, Transpose4by2) { TestTranspose(4, 2); }
TEST_F(SimdMatrixTest, Transpose4by3) { TestTranspose(4, 3); }
TEST_F(SimdMatrixTest, Transpose4by4) { TestTranspose(4, 4); }
TEST_F(SimdMatrixTest, Transpose4by5) { TestTranspose(4, 5); }
TEST_F(SimdMatrixTest, Transpose4by6) { TestTranspose(4, 6); }
TEST_F(SimdMatrixTest, Transpose4by7) { TestTranspose(4, 7); }
TEST_F(SimdMatrixTest, Transpose4by8) { TestTranspose(4, 8); }
TEST_F(SimdMatrixTest, Transpose8by1) { TestTranspose(8, 1); }
TEST_F(SimdMatrixTest, Transpose8by2) { TestTranspose(8, 2); }
TEST_F(SimdMatrixTest, Transpose8by3) { TestTranspose(8, 3); }
TEST_F(SimdMatrixTest, Transpose8by4) { TestTranspose(8, 4); }
TEST_F(SimdMatrixTest, Transpose8by5) { TestTranspose(8, 5); }
TEST_F(SimdMatrixTest, Transpose8by6) { TestTranspose(8, 6); }
TEST_F(SimdMatrixTest, Transpose8by7) { TestTranspose(8, 7); }
TEST_F(SimdMatrixTest, Transpose8by8) { TestTranspose(8, 8); }

// Transpose: rows non-divisible by 4
TEST_F(SimdMatrixTest, Transpose5by1) { TestTranspose(5, 1); }
TEST_F(SimdMatrixTest, Transpose5by2) { TestTranspose(5, 2); }
TEST_F(SimdMatrixTest, Transpose5by3) { TestTranspose(5, 3); }
TEST_F(SimdMatrixTest, Transpose5by4) { TestTranspose(5, 4); }
TEST_F(SimdMatrixTest, Transpose5by5) { TestTranspose(5, 5); }
TEST_F(SimdMatrixTest, Transpose5by6) { TestTranspose(5, 6); }
TEST_F(SimdMatrixTest, Transpose5by7) { TestTranspose(5, 7); }
TEST_F(SimdMatrixTest, Transpose5by8) { TestTranspose(5, 8); }
TEST_F(SimdMatrixTest, Transpose6by1) { TestTranspose(6, 1); }
TEST_F(SimdMatrixTest, Transpose6by2) { TestTranspose(6, 2); }
TEST_F(SimdMatrixTest, Transpose6by3) { TestTranspose(6, 3); }
TEST_F(SimdMatrixTest, Transpose6by4) { TestTranspose(6, 4); }
TEST_F(SimdMatrixTest, Transpose6by5) { TestTranspose(6, 5); }
TEST_F(SimdMatrixTest, Transpose6by6) { TestTranspose(6, 6); }
TEST_F(SimdMatrixTest, Transpose6by7) { TestTranspose(6, 7); }
TEST_F(SimdMatrixTest, Transpose6by8) { TestTranspose(6, 8); }
TEST_F(SimdMatrixTest, Transpose7by1) { TestTranspose(7, 1); }
TEST_F(SimdMatrixTest, Transpose7by2) { TestTranspose(7, 2); }
TEST_F(SimdMatrixTest, Transpose7by3) { TestTranspose(7, 3); }
TEST_F(SimdMatrixTest, Transpose7by4) { TestTranspose(7, 4); }
TEST_F(SimdMatrixTest, Transpose7by5) { TestTranspose(7, 5); }
TEST_F(SimdMatrixTest, Transpose7by6) { TestTranspose(7, 6); }
TEST_F(SimdMatrixTest, Transpose7by7) { TestTranspose(7, 7); }
TEST_F(SimdMatrixTest, Transpose7by8) { TestTranspose(7, 8); }

// Test method SimdMatrixTest::Product() used to generate test cases
TEST_F(SimdMatrixTest, Product) {
  uint rows = 4, cols1 = 3, cols2 = 2;
  vector<float> m  = { 1, 2, 3,
                       4, 5, 6,
                       7, 8, 9,
                       10, 11, 12 };
  vector<float> n = { 6, 5,
                      4, 3,
                      2, 1 };
  vector<float> mn = { 20, 14,
                       56, 41,
                       92, 68,
                       128, 95 };
  EXPECT_EQ(mn, Product(rows, cols1, cols2, m, n));
}

// Product: minimum rows1/cols1/cols2
TEST_F(SimdMatrixTest, Product1by4And4by1) { TestProduct(1, 4, 1); }

// Product: minimum cols1
TEST_F(SimdMatrixTest, Product1by4And4by2) { TestProduct(1, 4, 2); }
TEST_F(SimdMatrixTest, Product1by4And4by3) { TestProduct(1, 4, 3); }
TEST_F(SimdMatrixTest, Product2by4And4by1) { TestProduct(2, 4, 1); }
TEST_F(SimdMatrixTest, Product2by4And4by2) { TestProduct(2, 4, 2); }
TEST_F(SimdMatrixTest, Product2by4And4by3) { TestProduct(2, 4, 3); }
TEST_F(SimdMatrixTest, Product3by4And4by1) { TestProduct(3, 4, 1); }
TEST_F(SimdMatrixTest, Product3by4And4by2) { TestProduct(3, 4, 2); }
TEST_F(SimdMatrixTest, Product3by4And4by3) { TestProduct(3, 4, 3); }

// Product: rows1/cols1/cols2 divisible by 4
TEST_F(SimdMatrixTest, Product4by4And4by8) { TestProduct(4, 4, 8); }
TEST_F(SimdMatrixTest, Product4by8And8by4) { TestProduct(4, 8, 4); }
TEST_F(SimdMatrixTest, Product8by4And4by4) { TestProduct(8, 4, 4); }
TEST_F(SimdMatrixTest, Product4by8And8by8) { TestProduct(4, 8, 8); }
TEST_F(SimdMatrixTest, Product8by4And4by8) { TestProduct(8, 4, 8); }
TEST_F(SimdMatrixTest, Product8by8And8by8) { TestProduct(8, 8, 8); }

// Product: cols1/cols2 divisible by 4, rows1 non-divisible by 4
TEST_F(SimdMatrixTest, Product5by8And_by4)  { TestProduct(5, 4, 4); }
TEST_F(SimdMatrixTest, Product6by4And4by4)  { TestProduct(6, 4, 4); }
TEST_F(SimdMatrixTest, Product7by4And4by4)  { TestProduct(7, 4, 4); }
TEST_F(SimdMatrixTest, Product9by4And4by4)  { TestProduct(9, 4, 4); }
TEST_F(SimdMatrixTest, Product10by4And4by4) { TestProduct(10, 4, 4); }
TEST_F(SimdMatrixTest, Product11by4And4by4) { TestProduct(11, 4, 4); }
TEST_F(SimdMatrixTest, Product5by8And8by4)  { TestProduct(5, 8, 4); }
TEST_F(SimdMatrixTest, Product6by8And8by4)  { TestProduct(6, 8, 4); }
TEST_F(SimdMatrixTest, Product7by8And8by4)  { TestProduct(7, 8, 4); }
TEST_F(SimdMatrixTest, Product9by8And8by4)  { TestProduct(9, 8, 4); }
TEST_F(SimdMatrixTest, Product10by8And8by4) { TestProduct(10, 8, 4); }
TEST_F(SimdMatrixTest, Product11by8And8by4) { TestProduct(11, 8, 4); }
TEST_F(SimdMatrixTest, Product5by4And4by8)  { TestProduct(5, 4, 8); }
TEST_F(SimdMatrixTest, Product6by4And4by8)  { TestProduct(6, 4, 8); }
TEST_F(SimdMatrixTest, Product7by4And4by8)  { TestProduct(7, 4, 8); }
TEST_F(SimdMatrixTest, Product9by4And4by8)  { TestProduct(9, 4, 8); }
TEST_F(SimdMatrixTest, Product10by4And4by8) { TestProduct(10, 4, 8); }
TEST_F(SimdMatrixTest, Product11by4And4by8) { TestProduct(11, 4, 8); }
TEST_F(SimdMatrixTest, Product5by8And8by8)  { TestProduct(5, 8, 8); }
TEST_F(SimdMatrixTest, Product6by8And8by8)  { TestProduct(6, 8, 8); }
TEST_F(SimdMatrixTest, Product7by8And8by8)  { TestProduct(7, 8, 8); }
TEST_F(SimdMatrixTest, Product9by8And8by8)  { TestProduct(9, 8, 8); }
TEST_F(SimdMatrixTest, Product10by8And8by8) { TestProduct(10, 8, 8); }
TEST_F(SimdMatrixTest, Product11by8And8by8) { TestProduct(11, 8, 8); }

// Product: rows1/cols2 divisible by 4, cols1 non-divisible by 4
TEST_F(SimdMatrixTest, Product4by5And5by4)   { TestProduct(4, 5, 4); }
TEST_F(SimdMatrixTest, Product4by6And6by4)   { TestProduct(4, 6, 4); }
TEST_F(SimdMatrixTest, Product4by7And7by4)   { TestProduct(4, 7, 4); }
TEST_F(SimdMatrixTest, Product4by9And9by4)   { TestProduct(4, 9, 4); }
TEST_F(SimdMatrixTest, Product4by10And10by4) { TestProduct(4, 10, 4); }
TEST_F(SimdMatrixTest, Product4by11And11by4) { TestProduct(4, 11, 4); }
TEST_F(SimdMatrixTest, Product8by5And5by4)   { TestProduct(8, 5, 4); }
TEST_F(SimdMatrixTest, Product8by6And6by4)   { TestProduct(8, 6, 4); }
TEST_F(SimdMatrixTest, Product8by7And7by4)   { TestProduct(8, 7, 4); }
TEST_F(SimdMatrixTest, Product8by9And9by4)   { TestProduct(8, 9, 4); }
TEST_F(SimdMatrixTest, Product8by10And10by4) { TestProduct(8, 10, 4); }
TEST_F(SimdMatrixTest, Product8by11And11by4) { TestProduct(8, 11, 4); }
TEST_F(SimdMatrixTest, Product4by5And5by8)   { TestProduct(4, 5, 8); }
TEST_F(SimdMatrixTest, Product4by6And6by8)   { TestProduct(4, 6, 8); }
TEST_F(SimdMatrixTest, Product4by7And7by8)   { TestProduct(4, 7, 8); }
TEST_F(SimdMatrixTest, Product4by9And9by8)   { TestProduct(4, 9, 8); }
TEST_F(SimdMatrixTest, Product4by10And10by8) { TestProduct(4, 10, 8); }
TEST_F(SimdMatrixTest, Product4by11And11by8) { TestProduct(4, 11, 8); }
TEST_F(SimdMatrixTest, Product8by5And5by8)   { TestProduct(8, 5, 8); }
TEST_F(SimdMatrixTest, Product8by6And6by8)   { TestProduct(8, 6, 8); }
TEST_F(SimdMatrixTest, Product8by7And7by8)   { TestProduct(8, 7, 8); }
TEST_F(SimdMatrixTest, Product8by9And9by8)   { TestProduct(8, 9, 8); }
TEST_F(SimdMatrixTest, Product8by10And10by8) { TestProduct(8, 10, 8); }
TEST_F(SimdMatrixTest, Product8by11And11by8) { TestProduct(8, 11, 8); }

// Product: rows1/cols1 divisible by 4, cols2 non-divisible by 4
TEST_F(SimdMatrixTest, Product4by4And4by5)  { TestProduct(4, 4, 5); }
TEST_F(SimdMatrixTest, Product4by4And4by6)  { TestProduct(4, 4, 6); }
TEST_F(SimdMatrixTest, Product4by4And4by7)  { TestProduct(4, 4, 7); }
TEST_F(SimdMatrixTest, Product4by4And4by9)  { TestProduct(4, 4, 9); }
TEST_F(SimdMatrixTest, Product4by4And4by10) { TestProduct(4, 4, 10); }
TEST_F(SimdMatrixTest, Product4by4And4by11) { TestProduct(4, 4, 11); }
TEST_F(SimdMatrixTest, Product4by8And8by5)  { TestProduct(4, 8, 5); }
TEST_F(SimdMatrixTest, Product4by8And8by6)  { TestProduct(4, 8, 6); }
TEST_F(SimdMatrixTest, Product4by8And8by7)  { TestProduct(4, 8, 7); }
TEST_F(SimdMatrixTest, Product4by8And8by9)  { TestProduct(4, 8, 9); }
TEST_F(SimdMatrixTest, Product4by8And8by10) { TestProduct(4, 8, 10); }
TEST_F(SimdMatrixTest, Product4by8And8by11) { TestProduct(4, 8, 11); }
TEST_F(SimdMatrixTest, Product8by4And4by5)  { TestProduct(8, 4, 5); }
TEST_F(SimdMatrixTest, Product8by4And4by6)  { TestProduct(8, 4, 6); }
TEST_F(SimdMatrixTest, Product8by4And4by7)  { TestProduct(8, 4, 7); }
TEST_F(SimdMatrixTest, Product8by4And4by9)  { TestProduct(8, 4, 9); }
TEST_F(SimdMatrixTest, Product8by4And4by10) { TestProduct(8, 4, 10); }
TEST_F(SimdMatrixTest, Product8by4And4by11) { TestProduct(8, 4, 11); }
TEST_F(SimdMatrixTest, Product8by8And8by5)  { TestProduct(8, 8, 5); }
TEST_F(SimdMatrixTest, Product8by8And8by6)  { TestProduct(8, 8, 6); }
TEST_F(SimdMatrixTest, Product8by8And8by7)  { TestProduct(8, 8, 7); }
TEST_F(SimdMatrixTest, Product8by8And8by9)  { TestProduct(8, 8, 9); }
TEST_F(SimdMatrixTest, Product8by8And8by10) { TestProduct(8, 8, 10); }
TEST_F(SimdMatrixTest, Product8by8And8by11) { TestProduct(8, 8, 11); }

// Product: cols2 divisible by 4, rows1/cols1 non-divisible by 4
TEST_F(SimdMatrixTest, Product5by5And5by4)    { TestProduct(5, 5, 4); }
TEST_F(SimdMatrixTest, Product5by5And5by8)    { TestProduct(5, 5, 8); }
TEST_F(SimdMatrixTest, Product5by6And6by4)    { TestProduct(5, 6, 4); }
TEST_F(SimdMatrixTest, Product5by6And6by8)    { TestProduct(5, 6, 8); }
TEST_F(SimdMatrixTest, Product5by7And7by4)    { TestProduct(5, 7, 4); }
TEST_F(SimdMatrixTest, Product5by7And7by8)    { TestProduct(5, 7, 8); }
TEST_F(SimdMatrixTest, Product5by9And9by4)    { TestProduct(5, 9, 4); }
TEST_F(SimdMatrixTest, Product5by9And9by8)    { TestProduct(5, 9, 8); }
TEST_F(SimdMatrixTest, Product5by10And10by4)  { TestProduct(5, 10, 4); }
TEST_F(SimdMatrixTest, Product5by10And10by8)  { TestProduct(5, 10, 8); }
TEST_F(SimdMatrixTest, Product5by11And11by4)  { TestProduct(5, 11, 4); }
TEST_F(SimdMatrixTest, Product5by11And11by8)  { TestProduct(5, 11, 8); }
TEST_F(SimdMatrixTest, Product6by5And5by4)    { TestProduct(6, 5, 4); }
TEST_F(SimdMatrixTest, Product6by5And5by8)    { TestProduct(6, 5, 8); }
TEST_F(SimdMatrixTest, Product6by6And6by4)    { TestProduct(6, 6, 4); }
TEST_F(SimdMatrixTest, Product6by6And6by8)    { TestProduct(6, 6, 8); }
TEST_F(SimdMatrixTest, Product6by7And7by4)    { TestProduct(6, 7, 4); }
TEST_F(SimdMatrixTest, Product6by7And7by8)    { TestProduct(6, 7, 8); }
TEST_F(SimdMatrixTest, Product6by9And9by4)    { TestProduct(6, 9, 4); }
TEST_F(SimdMatrixTest, Product6by9And9by8)    { TestProduct(6, 9, 8); }
TEST_F(SimdMatrixTest, Product6by10And10by4)  { TestProduct(6, 10, 4); }
TEST_F(SimdMatrixTest, Product6by10And10by8)  { TestProduct(6, 10, 8); }
TEST_F(SimdMatrixTest, Product6by11And11by4)  { TestProduct(6, 11, 4); }
TEST_F(SimdMatrixTest, Product6by11And11by8)  { TestProduct(6, 11, 8); }
TEST_F(SimdMatrixTest, Product7by5And5by4)    { TestProduct(7, 5, 4); }
TEST_F(SimdMatrixTest, Product7by5And5by8)    { TestProduct(7, 5, 8); }
TEST_F(SimdMatrixTest, Product7by6And6by4)    { TestProduct(7, 6, 4); }
TEST_F(SimdMatrixTest, Product7by6And6by8)    { TestProduct(7, 6, 8); }
TEST_F(SimdMatrixTest, Product7by7And7by4)    { TestProduct(7, 7, 4); }
TEST_F(SimdMatrixTest, Product7by7And7by8)    { TestProduct(7, 7, 8); }
TEST_F(SimdMatrixTest, Product7by9And9by4)    { TestProduct(7, 9, 4); }
TEST_F(SimdMatrixTest, Product7by9And9by8)    { TestProduct(7, 9, 8); }
TEST_F(SimdMatrixTest, Product7by10And10by4)  { TestProduct(7, 10, 4); }
TEST_F(SimdMatrixTest, Product7by10And10by8)  { TestProduct(7, 10, 8); }
TEST_F(SimdMatrixTest, Product7by11And11by4)  { TestProduct(7, 11, 4); }
TEST_F(SimdMatrixTest, Product7by11And11by8)  { TestProduct(7, 11, 8); }
TEST_F(SimdMatrixTest, Product9by5And5by4)    { TestProduct(9, 5, 4); }
TEST_F(SimdMatrixTest, Product9by5And5by8)    { TestProduct(9, 5, 8); }
TEST_F(SimdMatrixTest, Product9by6And6by4)    { TestProduct(9, 6, 4); }
TEST_F(SimdMatrixTest, Product9by6And6by8)    { TestProduct(9, 6, 8); }
TEST_F(SimdMatrixTest, Product9by7And7by4)    { TestProduct(9, 7, 4); }
TEST_F(SimdMatrixTest, Product9by7And7by8)    { TestProduct(9, 7, 8); }
TEST_F(SimdMatrixTest, Product9by9And9by4)    { TestProduct(9, 9, 4); }
TEST_F(SimdMatrixTest, Product9by9And9by8)    { TestProduct(9, 9, 8); }
TEST_F(SimdMatrixTest, Product9by10And10by4)  { TestProduct(9, 10, 4); }
TEST_F(SimdMatrixTest, Product9by10And10by8)  { TestProduct(9, 10, 8); }
TEST_F(SimdMatrixTest, Product9by11And11by4)  { TestProduct(9, 11, 4); }
TEST_F(SimdMatrixTest, Product9by11And11by8)  { TestProduct(9, 11, 8); }
TEST_F(SimdMatrixTest, Product10by5And5by4)   { TestProduct(10, 5, 4); }
TEST_F(SimdMatrixTest, Product10by5And5by8)   { TestProduct(10, 5, 8); }
TEST_F(SimdMatrixTest, Product10by6And6by4)   { TestProduct(10, 6, 4); }
TEST_F(SimdMatrixTest, Product10by6And6by8)   { TestProduct(10, 6, 8); }
TEST_F(SimdMatrixTest, Product10by7And7by4)   { TestProduct(10, 7, 4); }
TEST_F(SimdMatrixTest, Product10by7And7by8)   { TestProduct(10, 7, 8); }
TEST_F(SimdMatrixTest, Product10by9And9by4)   { TestProduct(10, 9, 4); }
TEST_F(SimdMatrixTest, Product10by9And9by8)   { TestProduct(10, 9, 8); }
TEST_F(SimdMatrixTest, Product10by10And10by4) { TestProduct(10, 10, 4); }
TEST_F(SimdMatrixTest, Product10by10And10by8) { TestProduct(10, 10, 8); }
TEST_F(SimdMatrixTest, Product10by11And11by4) { TestProduct(10, 11, 4); }
TEST_F(SimdMatrixTest, Product10by11And11by8) { TestProduct(10, 11, 8); }
TEST_F(SimdMatrixTest, Product11by5And5by4)   { TestProduct(11, 5, 4); }
TEST_F(SimdMatrixTest, Product11by5And5by8)   { TestProduct(11, 5, 8); }
TEST_F(SimdMatrixTest, Product11by6And6by4)   { TestProduct(11, 6, 4); }
TEST_F(SimdMatrixTest, Product11by6And6by8)   { TestProduct(11, 6, 8); }
TEST_F(SimdMatrixTest, Product11by7And7by4)   { TestProduct(11, 7, 4); }
TEST_F(SimdMatrixTest, Product11by7And7by8)   { TestProduct(11, 7, 8); }
TEST_F(SimdMatrixTest, Product11by9And9by4)   { TestProduct(11, 9, 4); }
TEST_F(SimdMatrixTest, Product11by9And9by8)   { TestProduct(11, 9, 8); }
TEST_F(SimdMatrixTest, Product11by10And10by4) { TestProduct(11, 10, 4); }
TEST_F(SimdMatrixTest, Product11by10And10by8) { TestProduct(11, 10, 8); }
TEST_F(SimdMatrixTest, Product11by11And11by4) { TestProduct(11, 11, 4); }
TEST_F(SimdMatrixTest, Product11by11And11by8) { TestProduct(11, 11, 8); }

// Product: cols1 divisible by 4, rows1/cols2 non-divisible by 4
TEST_F(SimdMatrixTest, Product5by4And4by5)   { TestProduct(5, 4, 5); }
TEST_F(SimdMatrixTest, Product5by4And4by6)   { TestProduct(5, 4, 6); }
TEST_F(SimdMatrixTest, Product5by4And4by7)   { TestProduct(5, 4, 7); }
TEST_F(SimdMatrixTest, Product5by4And4by9)   { TestProduct(5, 4, 9); }
TEST_F(SimdMatrixTest, Product5by4And4by10)  { TestProduct(5, 4, 10); }
TEST_F(SimdMatrixTest, Product5by4And4by11)  { TestProduct(5, 4, 11); }
TEST_F(SimdMatrixTest, Product5by8And8by5)   { TestProduct(5, 8, 5); }
TEST_F(SimdMatrixTest, Product5by8And8by6)   { TestProduct(5, 8, 6); }
TEST_F(SimdMatrixTest, Product5by8And8by7)   { TestProduct(5, 8, 7); }
TEST_F(SimdMatrixTest, Product5by8And8by9)   { TestProduct(5, 8, 9); }
TEST_F(SimdMatrixTest, Product5by8And8by10)  { TestProduct(5, 8, 10); }
TEST_F(SimdMatrixTest, Product5by8And8by11)  { TestProduct(5, 8, 11); }
TEST_F(SimdMatrixTest, Product6by4And4by5)   { TestProduct(6, 4, 5); }
TEST_F(SimdMatrixTest, Product6by4And4by6)   { TestProduct(6, 4, 6); }
TEST_F(SimdMatrixTest, Product6by4And4by7)   { TestProduct(6, 4, 7); }
TEST_F(SimdMatrixTest, Product6by4And4by9)   { TestProduct(6, 4, 9); }
TEST_F(SimdMatrixTest, Product6by4And4by10)  { TestProduct(6, 4, 10); }
TEST_F(SimdMatrixTest, Product6by4And4by11)  { TestProduct(6, 4, 11); }
TEST_F(SimdMatrixTest, Product6by8And8by5)   { TestProduct(6, 8, 5); }
TEST_F(SimdMatrixTest, Product6by8And8by6)   { TestProduct(6, 8, 6); }
TEST_F(SimdMatrixTest, Product6by8And8by7)   { TestProduct(6, 8, 7); }
TEST_F(SimdMatrixTest, Product6by8And8by9)   { TestProduct(6, 8, 9); }
TEST_F(SimdMatrixTest, Product6by8And8by10)  { TestProduct(6, 8, 10); }
TEST_F(SimdMatrixTest, Product6by8And8by11)  { TestProduct(6, 8, 11); }
TEST_F(SimdMatrixTest, Product7by4And4by5)   { TestProduct(7, 4, 5); }
TEST_F(SimdMatrixTest, Product7by4And4by6)   { TestProduct(7, 4, 6); }
TEST_F(SimdMatrixTest, Product7by4And4by7)   { TestProduct(7, 4, 7); }
TEST_F(SimdMatrixTest, Product7by4And4by9)   { TestProduct(7, 4, 9); }
TEST_F(SimdMatrixTest, Product7by4And4by10)  { TestProduct(7, 4, 10); }
TEST_F(SimdMatrixTest, Product7by4And4by11)  { TestProduct(7, 4, 11); }
TEST_F(SimdMatrixTest, Product7by8And8by5)   { TestProduct(7, 8, 5); }
TEST_F(SimdMatrixTest, Product7by8And8by6)   { TestProduct(7, 8, 6); }
TEST_F(SimdMatrixTest, Product7by8And8by7)   { TestProduct(7, 8, 7); }
TEST_F(SimdMatrixTest, Product7by8And8by9)   { TestProduct(7, 8, 9); }
TEST_F(SimdMatrixTest, Product7by8And8by10)  { TestProduct(7, 8, 10); }
TEST_F(SimdMatrixTest, Product7by8And8by11)  { TestProduct(7, 8, 11); }
TEST_F(SimdMatrixTest, Product9by4And4by5)   { TestProduct(9, 4, 5); }
TEST_F(SimdMatrixTest, Product9by4And4by6)   { TestProduct(9, 4, 6); }
TEST_F(SimdMatrixTest, Product9by4And4by7)   { TestProduct(9, 4, 7); }
TEST_F(SimdMatrixTest, Product9by4And4by9)   { TestProduct(9, 4, 9); }
TEST_F(SimdMatrixTest, Product9by4And4by10)  { TestProduct(9, 4, 10); }
TEST_F(SimdMatrixTest, Product9by4And4by11)  { TestProduct(9, 4, 11); }
TEST_F(SimdMatrixTest, Product9by8And8by5)   { TestProduct(9, 8, 5); }
TEST_F(SimdMatrixTest, Product9by8And8by6)   { TestProduct(9, 8, 6); }
TEST_F(SimdMatrixTest, Product9by8And8by7)   { TestProduct(9, 8, 7); }
TEST_F(SimdMatrixTest, Product9by8And8by9)   { TestProduct(9, 8, 9); }
TEST_F(SimdMatrixTest, Product9by8And8by10)  { TestProduct(9, 8, 10); }
TEST_F(SimdMatrixTest, Product9by8And8by11)  { TestProduct(9, 8, 11); }
TEST_F(SimdMatrixTest, Product10by4And4by5)  { TestProduct(10, 4, 5); }
TEST_F(SimdMatrixTest, Product10by4And4by6)  { TestProduct(10, 4, 6); }
TEST_F(SimdMatrixTest, Product10by4And4by7)  { TestProduct(10, 4, 7); }
TEST_F(SimdMatrixTest, Product10by4And4by9)  { TestProduct(10, 4, 9); }
TEST_F(SimdMatrixTest, Product10by4And4by10) { TestProduct(10, 4, 10); }
TEST_F(SimdMatrixTest, Product10by4And4by11) { TestProduct(10, 4, 11); }
TEST_F(SimdMatrixTest, Product10by8And8by5)  { TestProduct(10, 8, 5); }
TEST_F(SimdMatrixTest, Product10by8And8by6)  { TestProduct(10, 8, 6); }
TEST_F(SimdMatrixTest, Product10by8And8by7)  { TestProduct(10, 8, 7); }
TEST_F(SimdMatrixTest, Product10by8And8by9)  { TestProduct(10, 8, 9); }
TEST_F(SimdMatrixTest, Product10by8And8by10) { TestProduct(10, 8, 10); }
TEST_F(SimdMatrixTest, Product10by8And8by11) { TestProduct(10, 8, 11); }
TEST_F(SimdMatrixTest, Product11by4And4by5)  { TestProduct(11, 4, 5); }
TEST_F(SimdMatrixTest, Product11by4And4by6)  { TestProduct(11, 4, 6); }
TEST_F(SimdMatrixTest, Product11by4And4by7)  { TestProduct(11, 4, 7); }
TEST_F(SimdMatrixTest, Product11by4And4by9)  { TestProduct(11, 4, 9); }
TEST_F(SimdMatrixTest, Product11by4And4by10) { TestProduct(11, 4, 10); }
TEST_F(SimdMatrixTest, Product11by4And4by11) { TestProduct(11, 4, 11); }
TEST_F(SimdMatrixTest, Product11by8And8by5)  { TestProduct(11, 8, 5); }
TEST_F(SimdMatrixTest, Product11by8And8by6)  { TestProduct(11, 8, 6); }
TEST_F(SimdMatrixTest, Product11by8And8by7)  { TestProduct(11, 8, 7); }
TEST_F(SimdMatrixTest, Product11by8And8by9)  { TestProduct(11, 8, 9); }
TEST_F(SimdMatrixTest, Product11by8And8by10) { TestProduct(11, 8, 10); }
TEST_F(SimdMatrixTest, Product11by8And8by11) { TestProduct(11, 8, 11); }

// Product: rows1 divisible by 4, cols1/cols2 non-divisible by 4
TEST_F(SimdMatrixTest, Product4by5And5by5)    { TestProduct(4, 5, 5); }
TEST_F(SimdMatrixTest, Product4by5And5by6)    { TestProduct(4, 5, 6); }
TEST_F(SimdMatrixTest, Product4by5And5by7)    { TestProduct(4, 5, 7); }
TEST_F(SimdMatrixTest, Product4by5And5by9)    { TestProduct(4, 5, 9); }
TEST_F(SimdMatrixTest, Product4by5And5by10)   { TestProduct(4, 5, 10); }
TEST_F(SimdMatrixTest, Product4by5And5by11)   { TestProduct(4, 5, 11); }
TEST_F(SimdMatrixTest, Product4by6And6by5)    { TestProduct(4, 6, 5); }
TEST_F(SimdMatrixTest, Product4by6And6by6)    { TestProduct(4, 6, 6); }
TEST_F(SimdMatrixTest, Product4by6And6by7)    { TestProduct(4, 6, 7); }
TEST_F(SimdMatrixTest, Product4by6And6by9)    { TestProduct(4, 6, 9); }
TEST_F(SimdMatrixTest, Product4by6And6by10)   { TestProduct(4, 6, 10); }
TEST_F(SimdMatrixTest, Product4by6And6by11)   { TestProduct(4, 6, 11); }
TEST_F(SimdMatrixTest, Product4by7And7by5)    { TestProduct(4, 7, 5); }
TEST_F(SimdMatrixTest, Product4by7And7by6)    { TestProduct(4, 7, 6); }
TEST_F(SimdMatrixTest, Product4by7And7by7)    { TestProduct(4, 7, 7); }
TEST_F(SimdMatrixTest, Product4by7And7by9)    { TestProduct(4, 7, 9); }
TEST_F(SimdMatrixTest, Product4by7And7by10)   { TestProduct(4, 7, 10); }
TEST_F(SimdMatrixTest, Product4by7And7by11)   { TestProduct(4, 7, 11); }
TEST_F(SimdMatrixTest, Product4by9And9by5)    { TestProduct(4, 9, 5); }
TEST_F(SimdMatrixTest, Product4by9And9by6)    { TestProduct(4, 9, 6); }
TEST_F(SimdMatrixTest, Product4by9And9by7)    { TestProduct(4, 9, 7); }
TEST_F(SimdMatrixTest, Product4by9And9by9)    { TestProduct(4, 9, 9); }
TEST_F(SimdMatrixTest, Product4by9And9by10)   { TestProduct(4, 9, 10); }
TEST_F(SimdMatrixTest, Product4by9And9by11)   { TestProduct(4, 9, 11); }
TEST_F(SimdMatrixTest, Product4by10And10by5)  { TestProduct(4, 10, 5); }
TEST_F(SimdMatrixTest, Product4by10And10by6)  { TestProduct(4, 10, 6); }
TEST_F(SimdMatrixTest, Product4by10And10by7)  { TestProduct(4, 10, 7); }
TEST_F(SimdMatrixTest, Product4by10And10by9)  { TestProduct(4, 10, 9); }
TEST_F(SimdMatrixTest, Product4by10And10by10) { TestProduct(4, 10, 10); }
TEST_F(SimdMatrixTest, Product4by10And10by11) { TestProduct(4, 10, 11); }
TEST_F(SimdMatrixTest, Product4by11And11by5)  { TestProduct(4, 11, 5); }
TEST_F(SimdMatrixTest, Product4by11And11by6)  { TestProduct(4, 11, 6); }
TEST_F(SimdMatrixTest, Product4by11And11by7)  { TestProduct(4, 11, 7); }
TEST_F(SimdMatrixTest, Product4by11And11by9)  { TestProduct(4, 11, 9); }
TEST_F(SimdMatrixTest, Product4by11And11by10) { TestProduct(4, 11, 10); }
TEST_F(SimdMatrixTest, Product4by11And11by11) { TestProduct(4, 11, 11); }
TEST_F(SimdMatrixTest, Product8by5And5by5)    { TestProduct(8, 5, 5); }
TEST_F(SimdMatrixTest, Product8by5And5by6)    { TestProduct(8, 5, 6); }
TEST_F(SimdMatrixTest, Product8by5And5by7)    { TestProduct(8, 5, 7); }
TEST_F(SimdMatrixTest, Product8by5And5by9)    { TestProduct(8, 5, 9); }
TEST_F(SimdMatrixTest, Product8by5And5by10)   { TestProduct(8, 5, 10); }
TEST_F(SimdMatrixTest, Product8by5And5by11)   { TestProduct(8, 5, 11); }
TEST_F(SimdMatrixTest, Product8by6And6by5)    { TestProduct(8, 6, 5); }
TEST_F(SimdMatrixTest, Product8by6And6by6)    { TestProduct(8, 6, 6); }
TEST_F(SimdMatrixTest, Product8by6And6by7)    { TestProduct(8, 6, 7); }
TEST_F(SimdMatrixTest, Product8by6And6by9)    { TestProduct(8, 6, 9); }
TEST_F(SimdMatrixTest, Product8by6And6by10)   { TestProduct(8, 6, 10); }
TEST_F(SimdMatrixTest, Product8by6And6by11)   { TestProduct(8, 6, 11); }
TEST_F(SimdMatrixTest, Product8by7And7by5)    { TestProduct(8, 7, 5); }
TEST_F(SimdMatrixTest, Product8by7And7by6)    { TestProduct(8, 7, 6); }
TEST_F(SimdMatrixTest, Product8by7And7by7)    { TestProduct(8, 7, 7); }
TEST_F(SimdMatrixTest, Product8by7And7by9)    { TestProduct(8, 7, 9); }
TEST_F(SimdMatrixTest, Product8by7And7by10)   { TestProduct(8, 7, 10); }
TEST_F(SimdMatrixTest, Product8by7And7by11)   { TestProduct(8, 7, 11); }
TEST_F(SimdMatrixTest, Product8by9And9by5)    { TestProduct(8, 9, 5); }
TEST_F(SimdMatrixTest, Product8by9And9by6)    { TestProduct(8, 9, 6); }
TEST_F(SimdMatrixTest, Product8by9And9by7)    { TestProduct(8, 9, 7); }
TEST_F(SimdMatrixTest, Product8by9And9by9)    { TestProduct(8, 9, 9); }
TEST_F(SimdMatrixTest, Product8by9And9by10)   { TestProduct(8, 9, 10); }
TEST_F(SimdMatrixTest, Product8by9And9by11)   { TestProduct(8, 9, 11); }
TEST_F(SimdMatrixTest, Product8by10And10by5)  { TestProduct(8, 10, 5); }
TEST_F(SimdMatrixTest, Product8by10And10by6)  { TestProduct(8, 10, 6); }
TEST_F(SimdMatrixTest, Product8by10And10by7)  { TestProduct(8, 10, 7); }
TEST_F(SimdMatrixTest, Product8by10And10by9)  { TestProduct(8, 10, 9); }
TEST_F(SimdMatrixTest, Product8by10And10by10) { TestProduct(8, 10, 10); }
TEST_F(SimdMatrixTest, Product8by10And10by11) { TestProduct(8, 10, 11); }
TEST_F(SimdMatrixTest, Product8by11And11by5)  { TestProduct(8, 11, 5); }
TEST_F(SimdMatrixTest, Product8by11And11by6)  { TestProduct(8, 11, 6); }
TEST_F(SimdMatrixTest, Product8by11And11by7)  { TestProduct(8, 11, 7); }
TEST_F(SimdMatrixTest, Product8by11And11by9)  { TestProduct(8, 11, 9); }
TEST_F(SimdMatrixTest, Product8by11And11by10) { TestProduct(8, 11, 10); }
TEST_F(SimdMatrixTest, Product8by11And11by11) { TestProduct(8, 11, 11); }

// Product: rows1/cols1/cols2 non-divisible by 4
TEST_F(SimdMatrixTest, Product5by5And5by5)     { TestProduct(5, 5, 5); }
TEST_F(SimdMatrixTest, Product5by5And5by6)     { TestProduct(5, 5, 6); }
TEST_F(SimdMatrixTest, Product5by5And5by7)     { TestProduct(5, 5, 7); }
TEST_F(SimdMatrixTest, Product5by5And5by9)     { TestProduct(5, 5, 9); }
TEST_F(SimdMatrixTest, Product5by5And5by10)    { TestProduct(5, 5, 10); }
TEST_F(SimdMatrixTest, Product5by5And5by11)    { TestProduct(5, 5, 11); }
TEST_F(SimdMatrixTest, Product5by6And6by5)     { TestProduct(5, 6, 5); }
TEST_F(SimdMatrixTest, Product5by6And6by6)     { TestProduct(5, 6, 6); }
TEST_F(SimdMatrixTest, Product5by6And6by7)     { TestProduct(5, 6, 7); }
TEST_F(SimdMatrixTest, Product5by6And6by9)     { TestProduct(5, 6, 9); }
TEST_F(SimdMatrixTest, Product5by6And6by10)    { TestProduct(5, 6, 10); }
TEST_F(SimdMatrixTest, Product5by6And6by11)    { TestProduct(5, 6, 11); }
TEST_F(SimdMatrixTest, Product5by7And7by5)     { TestProduct(5, 7, 5); }
TEST_F(SimdMatrixTest, Product5by7And7by6)     { TestProduct(5, 7, 6); }
TEST_F(SimdMatrixTest, Product5by7And7by7)     { TestProduct(5, 7, 7); }
TEST_F(SimdMatrixTest, Product5by7And7by9)     { TestProduct(5, 7, 9); }
TEST_F(SimdMatrixTest, Product5by7And7by10)    { TestProduct(5, 7, 10); }
TEST_F(SimdMatrixTest, Product5by7And7by11)    { TestProduct(5, 7, 11); }
TEST_F(SimdMatrixTest, Product5by9And9by5)     { TestProduct(5, 9, 5); }
TEST_F(SimdMatrixTest, Product5by9And9by6)     { TestProduct(5, 9, 6); }
TEST_F(SimdMatrixTest, Product5by9And9by7)     { TestProduct(5, 9, 7); }
TEST_F(SimdMatrixTest, Product5by9And9by9)     { TestProduct(5, 9, 9); }
TEST_F(SimdMatrixTest, Product5by9And9by10)    { TestProduct(5, 9, 10); }
TEST_F(SimdMatrixTest, Product5by9And9by11)    { TestProduct(5, 9, 11); }
TEST_F(SimdMatrixTest, Product5by10And10by5)   { TestProduct(5, 10, 5); }
TEST_F(SimdMatrixTest, Product5by10And10by6)   { TestProduct(5, 10, 6); }
TEST_F(SimdMatrixTest, Product5by10And10by7)   { TestProduct(5, 10, 7); }
TEST_F(SimdMatrixTest, Product5by10And10by9)   { TestProduct(5, 10, 9); }
TEST_F(SimdMatrixTest, Product5by10And10by10)  { TestProduct(5, 10, 10); }
TEST_F(SimdMatrixTest, Product5by10And10by11)  { TestProduct(5, 10, 11); }
TEST_F(SimdMatrixTest, Product5by11And11by5)   { TestProduct(5, 11, 5); }
TEST_F(SimdMatrixTest, Product5by11And11by6)   { TestProduct(5, 11, 6); }
TEST_F(SimdMatrixTest, Product5by11And11by7)   { TestProduct(5, 11, 7); }
TEST_F(SimdMatrixTest, Product5by11And11by9)   { TestProduct(5, 11, 9); }
TEST_F(SimdMatrixTest, Product5by11And11by10)  { TestProduct(5, 11, 10); }
TEST_F(SimdMatrixTest, Product5by11And11by11)  { TestProduct(5, 11, 11); }
TEST_F(SimdMatrixTest, Product6by5And5by5)     { TestProduct(6, 5, 5); }
TEST_F(SimdMatrixTest, Product6by5And5by6)     { TestProduct(6, 5, 6); }
TEST_F(SimdMatrixTest, Product6by5And5by7)     { TestProduct(6, 5, 7); }
TEST_F(SimdMatrixTest, Product6by5And5by9)     { TestProduct(6, 5, 9); }
TEST_F(SimdMatrixTest, Product6by5And5by10)    { TestProduct(6, 5, 10); }
TEST_F(SimdMatrixTest, Product6by5And5by11)    { TestProduct(6, 5, 11); }
TEST_F(SimdMatrixTest, Product6by6And6by5)     { TestProduct(6, 6, 5); }
TEST_F(SimdMatrixTest, Product6by6And6by6)     { TestProduct(6, 6, 6); }
TEST_F(SimdMatrixTest, Product6by6And6by7)     { TestProduct(6, 6, 7); }
TEST_F(SimdMatrixTest, Product6by6And6by9)     { TestProduct(6, 6, 9); }
TEST_F(SimdMatrixTest, Product6by6And6by10)    { TestProduct(6, 6, 10); }
TEST_F(SimdMatrixTest, Product6by6And6by11)    { TestProduct(6, 6, 11); }
TEST_F(SimdMatrixTest, Product6by7And7by5)     { TestProduct(6, 7, 5); }
TEST_F(SimdMatrixTest, Product6by7And7by6)     { TestProduct(6, 7, 6); }
TEST_F(SimdMatrixTest, Product6by7And7by7)     { TestProduct(6, 7, 7); }
TEST_F(SimdMatrixTest, Product6by7And7by9)     { TestProduct(6, 7, 9); }
TEST_F(SimdMatrixTest, Product6by7And7by10)    { TestProduct(6, 7, 10); }
TEST_F(SimdMatrixTest, Product6by7And7by11)    { TestProduct(6, 7, 11); }
TEST_F(SimdMatrixTest, Product6by9And9by5)     { TestProduct(6, 9, 5); }
TEST_F(SimdMatrixTest, Product6by9And9by6)     { TestProduct(6, 9, 6); }
TEST_F(SimdMatrixTest, Product6by9And9by7)     { TestProduct(6, 9, 7); }
TEST_F(SimdMatrixTest, Product6by9And9by9)     { TestProduct(6, 9, 9); }
TEST_F(SimdMatrixTest, Product6by9And9by10)    { TestProduct(6, 9, 10); }
TEST_F(SimdMatrixTest, Product6by9And9by11)    { TestProduct(6, 9, 11); }
TEST_F(SimdMatrixTest, Product6by10And10by5)   { TestProduct(6, 10, 5); }
TEST_F(SimdMatrixTest, Product6by10And10by6)   { TestProduct(6, 10, 6); }
TEST_F(SimdMatrixTest, Product6by10And10by7)   { TestProduct(6, 10, 7); }
TEST_F(SimdMatrixTest, Product6by10And10by9)   { TestProduct(6, 10, 9); }
TEST_F(SimdMatrixTest, Product6by10And10by10)  { TestProduct(6, 10, 10); }
TEST_F(SimdMatrixTest, Product6by10And10by11)  { TestProduct(6, 10, 11); }
TEST_F(SimdMatrixTest, Product6by11And11by5)   { TestProduct(6, 11, 5); }
TEST_F(SimdMatrixTest, Product6by11And11by6)   { TestProduct(6, 11, 6); }
TEST_F(SimdMatrixTest, Product6by11And11by7)   { TestProduct(6, 11, 7); }
TEST_F(SimdMatrixTest, Product6by11And11by9)   { TestProduct(6, 11, 9); }
TEST_F(SimdMatrixTest, Product6by11And11by10)  { TestProduct(6, 11, 10); }
TEST_F(SimdMatrixTest, Product6by11And11by11)  { TestProduct(6, 11, 11); }
TEST_F(SimdMatrixTest, Product7by5And5by5)     { TestProduct(7, 5, 5); }
TEST_F(SimdMatrixTest, Product7by5And5by6)     { TestProduct(7, 5, 6); }
TEST_F(SimdMatrixTest, Product7by5And5by7)     { TestProduct(7, 5, 7); }
TEST_F(SimdMatrixTest, Product7by5And5by9)     { TestProduct(7, 5, 9); }
TEST_F(SimdMatrixTest, Product7by5And5by10)    { TestProduct(7, 5, 10); }
TEST_F(SimdMatrixTest, Product7by5And5by11)    { TestProduct(7, 5, 11); }
TEST_F(SimdMatrixTest, Product7by6And6by5)     { TestProduct(7, 6, 5); }
TEST_F(SimdMatrixTest, Product7by6And6by6)     { TestProduct(7, 6, 6); }
TEST_F(SimdMatrixTest, Product7by6And6by7)     { TestProduct(7, 6, 7); }
TEST_F(SimdMatrixTest, Product7by6And6by9)     { TestProduct(7, 6, 9); }
TEST_F(SimdMatrixTest, Product7by6And6by10)    { TestProduct(7, 6, 10); }
TEST_F(SimdMatrixTest, Product7by6And6by11)    { TestProduct(7, 6, 11); }
TEST_F(SimdMatrixTest, Product7by7And7by5)     { TestProduct(7, 7, 5); }
TEST_F(SimdMatrixTest, Product7by7And7by6)     { TestProduct(7, 7, 6); }
TEST_F(SimdMatrixTest, Product7by7And7by7)     { TestProduct(7, 7, 7); }
TEST_F(SimdMatrixTest, Product7by7And7by9)     { TestProduct(7, 7, 9); }
TEST_F(SimdMatrixTest, Product7by7And7by10)    { TestProduct(7, 7, 10); }
TEST_F(SimdMatrixTest, Product7by7And7by11)    { TestProduct(7, 7, 11); }
TEST_F(SimdMatrixTest, Product7by9And9by5)     { TestProduct(7, 9, 5); }
TEST_F(SimdMatrixTest, Product7by9And9by6)     { TestProduct(7, 9, 6); }
TEST_F(SimdMatrixTest, Product7by9And9by7)     { TestProduct(7, 9, 7); }
TEST_F(SimdMatrixTest, Product7by9And9by9)     { TestProduct(7, 9, 9); }
TEST_F(SimdMatrixTest, Product7by9And9by10)    { TestProduct(7, 9, 10); }
TEST_F(SimdMatrixTest, Product7by9And9by11)    { TestProduct(7, 9, 11); }
TEST_F(SimdMatrixTest, Product7by10And10by5)   { TestProduct(7, 10, 5); }
TEST_F(SimdMatrixTest, Product7by10And10by6)   { TestProduct(7, 10, 6); }
TEST_F(SimdMatrixTest, Product7by10And10by7)   { TestProduct(7, 10, 7); }
TEST_F(SimdMatrixTest, Product7by10And10by9)   { TestProduct(7, 10, 9); }
TEST_F(SimdMatrixTest, Product7by10And10by10)  { TestProduct(7, 10, 10); }
TEST_F(SimdMatrixTest, Product7by10And10by11)  { TestProduct(7, 10, 11); }
TEST_F(SimdMatrixTest, Product7by11And11by5)   { TestProduct(7, 11, 5); }
TEST_F(SimdMatrixTest, Product7by11And11by6)   { TestProduct(7, 11, 6); }
TEST_F(SimdMatrixTest, Product7by11And11by7)   { TestProduct(7, 11, 7); }
TEST_F(SimdMatrixTest, Product7by11And11by9)   { TestProduct(7, 11, 9); }
TEST_F(SimdMatrixTest, Product7by11And11by10)  { TestProduct(7, 11, 10); }
TEST_F(SimdMatrixTest, Product7by11And11by11)  { TestProduct(7, 11, 11); }
TEST_F(SimdMatrixTest, Product9by5And5by5)     { TestProduct(9, 5, 5); }
TEST_F(SimdMatrixTest, Product9by5And5by6)     { TestProduct(9, 5, 6); }
TEST_F(SimdMatrixTest, Product9by5And5by7)     { TestProduct(9, 5, 7); }
TEST_F(SimdMatrixTest, Product9by5And5by9)     { TestProduct(9, 5, 9); }
TEST_F(SimdMatrixTest, Product9by5And5by10)    { TestProduct(9, 5, 10); }
TEST_F(SimdMatrixTest, Product9by5And5by11)    { TestProduct(9, 5, 11); }
TEST_F(SimdMatrixTest, Product9by6And6by5)     { TestProduct(9, 6, 5); }
TEST_F(SimdMatrixTest, Product9by6And6by6)     { TestProduct(9, 6, 6); }
TEST_F(SimdMatrixTest, Product9by6And6by7)     { TestProduct(9, 6, 7); }
TEST_F(SimdMatrixTest, Product9by6And6by9)     { TestProduct(9, 6, 9); }
TEST_F(SimdMatrixTest, Product9by6And6by10)    { TestProduct(9, 6, 10); }
TEST_F(SimdMatrixTest, Product9by6And6by11)    { TestProduct(9, 6, 11); }
TEST_F(SimdMatrixTest, Product9by7And7by5)     { TestProduct(9, 7, 5); }
TEST_F(SimdMatrixTest, Product9by7And7by6)     { TestProduct(9, 7, 6); }
TEST_F(SimdMatrixTest, Product9by7And7by7)     { TestProduct(9, 7, 7); }
TEST_F(SimdMatrixTest, Product9by7And7by9)     { TestProduct(9, 7, 9); }
TEST_F(SimdMatrixTest, Product9by7And7by10)    { TestProduct(9, 7, 10); }
TEST_F(SimdMatrixTest, Product9by7And7by11)    { TestProduct(9, 7, 11); }
TEST_F(SimdMatrixTest, Product9by9And9by5)     { TestProduct(9, 9, 5); }
TEST_F(SimdMatrixTest, Product9by9And9by6)     { TestProduct(9, 9, 6); }
TEST_F(SimdMatrixTest, Product9by9And9by7)     { TestProduct(9, 9, 7); }
TEST_F(SimdMatrixTest, Product9by9And9by9)     { TestProduct(9, 9, 9); }
TEST_F(SimdMatrixTest, Product9by9And9by10)    { TestProduct(9, 9, 10); }
TEST_F(SimdMatrixTest, Product9by9And9by11)    { TestProduct(9, 9, 11); }
TEST_F(SimdMatrixTest, Product9by10And10by5)   { TestProduct(9, 10, 5); }
TEST_F(SimdMatrixTest, Product9by10And10by6)   { TestProduct(9, 10, 6); }
TEST_F(SimdMatrixTest, Product9by10And10by7)   { TestProduct(9, 10, 7); }
TEST_F(SimdMatrixTest, Product9by10And10by9)   { TestProduct(9, 10, 9); }
TEST_F(SimdMatrixTest, Product9by10And10by10)  { TestProduct(9, 10, 10); }
TEST_F(SimdMatrixTest, Product9by10And10by11)  { TestProduct(9, 10, 11); }
TEST_F(SimdMatrixTest, Product9by11And11by5)   { TestProduct(9, 11, 5); }
TEST_F(SimdMatrixTest, Product9by11And11by6)   { TestProduct(9, 11, 6); }
TEST_F(SimdMatrixTest, Product9by11And11by7)   { TestProduct(9, 11, 7); }
TEST_F(SimdMatrixTest, Product9by11And11by9)   { TestProduct(9, 11, 9); }
TEST_F(SimdMatrixTest, Product9by11And11by10)  { TestProduct(9, 11, 10); }
TEST_F(SimdMatrixTest, Product9by11And11by11)  { TestProduct(9, 11, 11); }
TEST_F(SimdMatrixTest, Product10by5And5by5)    { TestProduct(10, 5, 5); }
TEST_F(SimdMatrixTest, Product10by5And5by6)    { TestProduct(10, 5, 6); }
TEST_F(SimdMatrixTest, Product10by5And5by7)    { TestProduct(10, 5, 7); }
TEST_F(SimdMatrixTest, Product10by5And5by9)    { TestProduct(10, 5, 9); }
TEST_F(SimdMatrixTest, Product10by5And5by10)   { TestProduct(10, 5, 10); }
TEST_F(SimdMatrixTest, Product10by5And5by11)   { TestProduct(10, 5, 11); }
TEST_F(SimdMatrixTest, Product10by6And6by5)    { TestProduct(10, 6, 5); }
TEST_F(SimdMatrixTest, Product10by6And6by6)    { TestProduct(10, 6, 6); }
TEST_F(SimdMatrixTest, Product10by6And6by7)    { TestProduct(10, 6, 7); }
TEST_F(SimdMatrixTest, Product10by6And6by9)    { TestProduct(10, 6, 9); }
TEST_F(SimdMatrixTest, Product10by6And6by10)   { TestProduct(10, 6, 10); }
TEST_F(SimdMatrixTest, Product10by6And6by11)   { TestProduct(10, 6, 11); }
TEST_F(SimdMatrixTest, Product10by7And7by5)    { TestProduct(10, 7, 5); }
TEST_F(SimdMatrixTest, Product10by7And7by6)    { TestProduct(10, 7, 6); }
TEST_F(SimdMatrixTest, Product10by7And7by7)    { TestProduct(10, 7, 7); }
TEST_F(SimdMatrixTest, Product10by7And7by9)    { TestProduct(10, 7, 9); }
TEST_F(SimdMatrixTest, Product10by7And7by10)   { TestProduct(10, 7, 10); }
TEST_F(SimdMatrixTest, Product10by7And7by11)   { TestProduct(10, 7, 11); }
TEST_F(SimdMatrixTest, Product10by9And9by5)    { TestProduct(10, 9, 5); }
TEST_F(SimdMatrixTest, Product10by9And9by6)    { TestProduct(10, 9, 6); }
TEST_F(SimdMatrixTest, Product10by9And9by7)    { TestProduct(10, 9, 7); }
TEST_F(SimdMatrixTest, Product10by9And9by9)    { TestProduct(10, 9, 9); }
TEST_F(SimdMatrixTest, Product10by9And9by10)   { TestProduct(10, 9, 10); }
TEST_F(SimdMatrixTest, Product10by9And9by11)   { TestProduct(10, 9, 11); }
TEST_F(SimdMatrixTest, Product10by10And10by5)  { TestProduct(10, 10, 5); }
TEST_F(SimdMatrixTest, Product10by10And10by6)  { TestProduct(10, 10, 6); }
TEST_F(SimdMatrixTest, Product10by10And10by7)  { TestProduct(10, 10, 7); }
TEST_F(SimdMatrixTest, Product10by10And10by9)  { TestProduct(10, 10, 9); }
TEST_F(SimdMatrixTest, Product10by10And10by10) { TestProduct(10, 10, 10); }
TEST_F(SimdMatrixTest, Product10by10And10by11) { TestProduct(10, 10, 11); }
TEST_F(SimdMatrixTest, Product10by11And11by5)  { TestProduct(10, 11, 5); }
TEST_F(SimdMatrixTest, Product10by11And11by6)  { TestProduct(10, 11, 6); }
TEST_F(SimdMatrixTest, Product10by11And11by7)  { TestProduct(10, 11, 7); }
TEST_F(SimdMatrixTest, Product10by11And11by9)  { TestProduct(10, 11, 9); }
TEST_F(SimdMatrixTest, Product10by11And11by10) { TestProduct(10, 11, 10); }
TEST_F(SimdMatrixTest, Product10by11And11by11) { TestProduct(10, 11, 11); }
TEST_F(SimdMatrixTest, Product11by5And5by5)    { TestProduct(11, 5, 5); }
TEST_F(SimdMatrixTest, Product11by5And5by6)    { TestProduct(11, 5, 6); }
TEST_F(SimdMatrixTest, Product11by5And5by7)    { TestProduct(11, 5, 7); }
TEST_F(SimdMatrixTest, Product11by5And5by9)    { TestProduct(11, 5, 9); }
TEST_F(SimdMatrixTest, Product11by5And5by10)   { TestProduct(11, 5, 10); }
TEST_F(SimdMatrixTest, Product11by5And5by11)   { TestProduct(11, 5, 11); }
TEST_F(SimdMatrixTest, Product11by6And6by5)    { TestProduct(11, 6, 5); }
TEST_F(SimdMatrixTest, Product11by6And6by6)    { TestProduct(11, 6, 6); }
TEST_F(SimdMatrixTest, Product11by6And6by7)    { TestProduct(11, 6, 7); }
TEST_F(SimdMatrixTest, Product11by6And6by9)    { TestProduct(11, 6, 9); }
TEST_F(SimdMatrixTest, Product11by6And6by10)   { TestProduct(11, 6, 10); }
TEST_F(SimdMatrixTest, Product11by6And6by11)   { TestProduct(11, 6, 11); }
TEST_F(SimdMatrixTest, Product11by7And7by5)    { TestProduct(11, 7, 5); }
TEST_F(SimdMatrixTest, Product11by7And7by6)    { TestProduct(11, 7, 6); }
TEST_F(SimdMatrixTest, Product11by7And7by7)    { TestProduct(11, 7, 7); }
TEST_F(SimdMatrixTest, Product11by7And7by9)    { TestProduct(11, 7, 9); }
TEST_F(SimdMatrixTest, Product11by7And7by10)   { TestProduct(11, 7, 10); }
TEST_F(SimdMatrixTest, Product11by7And7by11)   { TestProduct(11, 7, 11); }
TEST_F(SimdMatrixTest, Product11by9And9by5)    { TestProduct(11, 9, 5); }
TEST_F(SimdMatrixTest, Product11by9And9by6)    { TestProduct(11, 9, 6); }
TEST_F(SimdMatrixTest, Product11by9And9by7)    { TestProduct(11, 9, 7); }
TEST_F(SimdMatrixTest, Product11by9And9by9)    { TestProduct(11, 9, 9); }
TEST_F(SimdMatrixTest, Product11by9And9by10)   { TestProduct(11, 9, 10); }
TEST_F(SimdMatrixTest, Product11by9And9by11)   { TestProduct(11, 9, 11); }
TEST_F(SimdMatrixTest, Product11by10And10by5)  { TestProduct(11, 10, 5); }
TEST_F(SimdMatrixTest, Product11by10And10by6)  { TestProduct(11, 10, 6); }
TEST_F(SimdMatrixTest, Product11by10And10by7)  { TestProduct(11, 10, 7); }
TEST_F(SimdMatrixTest, Product11by10And10by9)  { TestProduct(11, 10, 9); }
TEST_F(SimdMatrixTest, Product11by10And10by10) { TestProduct(11, 10, 10); }
TEST_F(SimdMatrixTest, Product11by10And10by11) { TestProduct(11, 10, 11); }
TEST_F(SimdMatrixTest, Product11by11And11by5)  { TestProduct(11, 11, 5); }
TEST_F(SimdMatrixTest, Product11by11And11by6)  { TestProduct(11, 11, 6); }
TEST_F(SimdMatrixTest, Product11by11And11by7)  { TestProduct(11, 11, 7); }
TEST_F(SimdMatrixTest, Product11by11And11by9)  { TestProduct(11, 11, 9); }
TEST_F(SimdMatrixTest, Product11by11And11by10) { TestProduct(11, 11, 10); }
TEST_F(SimdMatrixTest, Product11by11And11by11) { TestProduct(11, 11, 11); }
