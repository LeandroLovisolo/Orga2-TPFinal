#ifndef __SIMD_MATRIX_H__
#define __SIMD_MATRIX_H__

extern "C" {

typedef unsigned int uint;

// In-place scalar product
void simd_matrix_scalar_product(uint size, float *m, float c);

// In-place addition (m = m + n)
void simd_matrix_addition(uint size, float *m, const float *n);

// In-place subtraction (m = m - n)
void simd_matrix_subtraction(uint size, float *m, const float *n);

// In-place coefficient-wise product (m = m .* n)
void simd_matrix_coeff_wise_product(uint size, float *m, const float *n);

// Matrix transposition
void simd_matrix_transpose(uint rows, uint cols, const float *m, float* n);

// Matrix product
void simd_matrix_product(uint rows1, uint cols1, uint cols2,
                         const float* m, const float* nt, float* p);

}  // extern "C"

#endif  // __SIMD_MATRIX_H__
