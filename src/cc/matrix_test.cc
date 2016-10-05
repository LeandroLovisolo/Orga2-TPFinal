#include <memory>

#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "matrix.h"

using namespace std;

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
