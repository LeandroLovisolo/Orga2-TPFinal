global simd_matrix_scalar_product
global simd_matrix_addition
global simd_matrix_subtraction
global simd_matrix_coeff_wise_product
global simd_matrix_transpose
global simd_matrix_product

section .text

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; MACRO coeff_wise_vector_to_vector_op 1       ;;
;;                                              ;;
;; Generates the body of a C-compatible         ;;
;; function implementing a coefficient-wise,    ;;
;; vector-to-vector, single precision floating  ;;
;; point SIMD operation such as                 ;;
;; coefficient-wise addition, subtraction and   ;;
;; multiplication.                              ;;
;;                                              ;;
;; The macro takes a single parameter           ;;
;; indicating the SIMD instruction to be used.  ;;
;; Example invocation:                          ;;
;;                                              ;;
;;   coeff_wise_vector_to_vector_op addps       ;;
;;                                              ;;
;; The generated function expects 3 arguments:  ;;
;;  - vector size (unsigned integer)            ;;
;;  - pointer to first vector                   ;;
;;  - pointer to second vector                  ;;
;;                                              ;;
;; The following example illustrates the        ;;
;; signature of a C function generated with     ;;
;; this macro:                                  ;;
;;                                              ;;
;;   void coeff_wise_add(uint size,  // rdi     ;;
;;                       float* m,   // rsi     ;;
;;                       float* n)   // rdx     ;;
;;                                              ;;
;; The corresponding assembly code can be       ;;
;; generated with:                              ;;
;;                                              ;;
;;   coeff_wise_add:                            ;;
;;     coeff_wise_vector_to_vector_op addps     ;;
;;                                              ;;
;; Note: generated code assumes size >= 4.      ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

%macro coeff_wise_vector_to_vector_op 1
  push rbp
  mov rbp, rsp
  push rbx
  push r12
  push r13
  push r14
  push r15

  ; Clear high 32 bits of rdi
  mov edi, edi

  ; Store in r12 the maximum offset we want use with SIMD the instruction
  mov rax, rdi              ; rax = size
  sub rax, 4                ; rax = size - 4
  mov rbx, 4                ; rbx = 4
  push rdx
  mul rbx                   ; rax = 4 * (size - 4)
  pop rdx
  mov r12, rax              ; r12 = 4 * (size - 4)

  ; Initialize r13 as the offset register used in main loop
  xor r13, r13              ; r13 = 0

  ; Main loop
%?_%1_loop:
  movups xmm0, [rsi + r13]  ; Copy values at current offset in vector v
  movups xmm1, [rdx + r13]  ; Copy values at current offset in vector w
  %1 xmm0, xmm1             ; Execute SIMD op
  movups [rsi + r13], xmm0  ; Store results at current offset in vector v
  movups xmm2, xmm0         ; Backup xmm0 in xmm2 (see case 3 below)
  mov r14, r13              ; Backup offset register r13 in r14

  ; Compute difference between current and maximum offset
  mov r15, r12              ; r15 = max offset
  sub r15, r13              ; r15 = max offset - current offset

  ; Case 1: current offset = max offset
  jz exit_%?_%1_loop

  ; Case 2: 4 or more elements left
  cmp r15, 16
  jge %?_%1_loop_next

  ; Case 3: 1, 2 or 3 elements left
  movups xmm0, [rsi + r12]  ; Copy values at max offset in vector v
  movups xmm1, [rdx + r12]  ; Copy values at max offset in vector w
  %1 xmm0, xmm1             ; Execute SIMD op
  movups [rsi + r12], xmm0  ; Store results at max offset in vector v
  movups [rsi + r14], xmm2  ; Restore backed-up values
  jmp exit_%?_%1_loop

%?_%1_loop_next:
  add r13, 16               ; Advance 4 elements
  jmp %?_%1_loop

exit_%?_%1_loop:
  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  ret
%endmacro

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void simd_matrix_scalar_product(uint size,  // rdi  ;;
;;                                 float* m,   // rsi  ;;
;;                                 float c)    // xmm0 ;;
;; Assumes size >= 4.                                  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

simd_matrix_scalar_product:
  push rbp
  mov rbp, rsp
  push rbx
  push r12
  push r13
  push r14
  push r15

  ; Clear high 32 bits of rdi
  mov edi, edi

  ; Replicate value of xmm0
  pshufd xmm0, xmm0, 0      ; xmm0 = c | c | c | c

  ; Store in r12 the maximum offset we want use with mulps instruction
  mov rax, rdi              ; rax = size
  sub rax, 4                ; rax = size - 4
  mov rbx, 4                ; rbx = 4
  push rdx
  mul rbx                   ; rax = 4 * (size - 4)
  pop rdx
  mov r12, rax              ; r12 = 4 * (size - 4)

  ; Initialize r13 as the offset register used in scalar product loop
  xor r13, r13              ; r13 = 0

  ; Main loop
scalar_product_loop:
  movups xmm1, [rsi + r13]  ; Copy values at current offset in matrix m
  mulps xmm1, xmm0          ; Multiply values
  movups [rsi + r13], xmm1  ; Store results at current offset in matrix m
  movups xmm2, xmm1         ; Backup xmm0 in xmm2 (see case 3 below)
  mov r14, r13              ; Backup offset register r13 in r14

  ; Compute difference between current and maximum offset
  mov r15, r12              ; r15 = max offset
  sub r15, r13              ; r15 = max offset - current offset

  ; Case 1: current offset = max offset
  jz exit_scalar_product_loop

  ; Case 2: 4 or more elements left
  cmp r15, 16
  jge scalar_product_loop_next

  ; Case 3: 1, 2 or 3 elements left
  movups xmm1, [rsi + r12]  ; Copy values at max offset in matrix m
  mulps xmm1, xmm0          ; Add up values
  movups [rsi + r12], xmm1  ; Store results at max offset in matrix m
  movups [rsi + r14], xmm2  ; Restore backed-up values
  jmp exit_scalar_product_loop

scalar_product_loop_next:
  add r13, 16               ; Advance 4 elements
  jmp scalar_product_loop

exit_scalar_product_loop:
  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void simd_matrix_addition(uint size,       // rdi ;;
;;                           float* m,        // rsi ;;
;;                           const float* n)  // rdx ;;
;; Assumes size >= 4.                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

simd_matrix_addition:
  coeff_wise_vector_to_vector_op addps

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void simd_matrix_subtraction(uint size,    // rdi ;;
;;                           float* m,        // rsi ;;
;;                           const float* n)  // rdx ;;
;; Assumes size >= 4.                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

simd_matrix_subtraction:
  coeff_wise_vector_to_vector_op subps

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void simd_matrix_coeff_wise_product(uint size,      // rdi ;;
;;                                    float* m,        // rsi ;;
;;                                    const float* n)  // rdx ;;
;; Assumes size >= 4.                                         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

simd_matrix_coeff_wise_product:
  coeff_wise_vector_to_vector_op mulps

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void simd_matrix_transpose(uint rows,       // rdi ;;
;;                            uint cols,       // rsi ;;
;;                            const float* m,  // rdx ;;
;;                            float* n)        // rcx ;;
;; Assumes rows >= 4.                                 ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

simd_matrix_transpose:
  push rbp
  mov rbp, rsp
  push rbx
  push r12
  push r13
  push r14
  push r15

  ; Clear high 32 bits of rdi and rsi
  mov edi, edi
  mov esi, esi

  ; Column index (j)
  xor r13, r13             ; r13 = 0

  ; Iterate over columns
transpose_cols_loop:

  ; Row index (i)
  xor r12, r12             ; r12 = 0

  ; Iterate over rows
transpose_rows_loop:

  ; Copy m[i..i+3, j] to n[j, i..i+3]
  call transpose_ij

  ; Compute number of remaining rows left
  mov r14, rdi             ; r14 = rows
  sub r14, r12             ; r14 = rows - current row
  sub r14, 4               ; r14 = rows - current row - 4

  ; Case 1: no rows left
  jz exit_transpose_rows_loop

  ; Case 2: 4 or more rows left
  cmp r14, 4
  jge transpose_rows_loop_next

  ; Case 3: 1, 2 or 3 rows left
  mov r12, rdi             ; i = r12 = rows
  sub r12, 4               ; i = r12 = rows - 4
  call transpose_ij
  jmp exit_transpose_rows_loop

transpose_rows_loop_next:
  add r12, 4               ; Advance 4 rows
  jmp transpose_rows_loop

exit_transpose_rows_loop:
  ; Increase column index and iterate
  inc r13
  cmp r13, rsi
  jne transpose_cols_loop

  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure transpose_ij copies m[i..i+3, j] to n[j, i..i+3].     ;;
;; Expects rdi=rows, rsi=cols, rdx=pointer to m, rcx=pointer to n, ;;
;; r12=i, r13=j.                                                   ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

transpose_ij:

  ; Compute offset for m(i, j)
  mov rax, r12             ; rax = i
  push rdx
  mul rsi                  ; rax = i * cols
  add rax, r13             ; rax = (i * cols) + j
  mov rbx, 4               ; rbx = 4
  mul rbx                  ; rax = 4 * ((i * cols) + j)
  pop rdx

  ; Store m(i, j) in xmm0[0]
  movss xmm0, [rdx + rax]  ; xmm0 = 00000000  | 00000000 | 00000000 | m(i, j)

  ; Compute offset for m(i + 1, j)
  mov rax, r12             ; rax = i
  inc rax                  ; rax = i + 1
  push rdx
  mul rsi                  ; rax = (i + 1) * cols
  add rax, r13             ; rax = ((i + 1) * cols) + j
  mov rbx, 4               ; rbx = 4
  mul rbx                  ; rax = 4 * (((i + 1) * cols) + j)
  pop rdx

  ; Store m(i + 1, j) in xmm1[1]
  movss xmm1, [rdx + rax]  ; xmm1 = 00000000 | 00000000 | 00000000 | m(i+1,j)
  pshufd xmm1, xmm1, 0xE1  ; xmm1 = 00000000 | 00000000 | m(i+1,j) | 00000000
                           ; order = 11 10 00 01

  ; Compute offset for m(i + 2, j)
  mov rax, r12             ; rax = i
  add rax, 2               ; rax = i + 2
  push rdx
  mul rsi                  ; rax = (i + 2) * cols
  add rax, r13             ; rax = ((i + 2) * cols) + j
  mov rbx, 4               ; rbx = 4
  mul rbx                  ; rax = 4 * (((i + 2) * cols) + j)
  pop rdx

  ; Store m(i + 2, j) in xmm2[2]
  movss xmm2, [rdx + rax]  ; xmm2 = 00000000 | 00000000 | 00000000 | m(i+2,j)
  pshufd xmm2, xmm2, 0xC6  ; xmm2 = 00000000 | m(i+2,j) | 00000000 | 00000000
                           ; order = 11 00 01 10

  ; Compute offset for m(i + 3, j)
  mov rax, r12             ; rax = i
  add rax, 3               ; rax = i + 3
  push rdx
  mul rsi                  ; rax = (i + 3) * cols
  add rax, r13             ; rax = ((i + 3) * cols) + j
  mov rbx, 4               ; rbx = 4
  mul rbx                  ; rax = 4 * (((i + 3) * cols) + j)
  pop rdx

  ; Store m(i + 3, j) in xmm3[3]
  movss xmm3, [rdx + rax]  ; xmm3 = 00000000 | 00000000 | 00000000 | m(i+3,j)
  pshufd xmm3, xmm3, 0x27  ; xmm3 = m(i+3,j) | 00000000 | 00000000 | 00000000
                           ; order = 00 10 01 11

  ; Merge values in one single XMM register
  addps xmm0, xmm1         ; xmm0 = 00000000 | 00000000 | m(i+1,j) |   m(i,j)
  addps xmm0, xmm2         ; xmm0 = 00000000 | m(i+2,j) | m(i+1,j) |   m(i,j)
  addps xmm0, xmm3         ; xmm0 = m(i+3,j) | m(i+2,j) | m(i+1,j) |   m(i,j)

  ; Compute offset for n(j, i)
  mov rax, r13             ; rax = j
  push rdx
  mul rdi                  ; rax = j * rows
  add rax, r12             ; rax = (j * rows) + i
  mov rbx, 4               ; rbx = 4
  mul rbx                  ; rax = 4 * ((j * rows) + i)
  pop rdx

  ; Copy contents of xmm0 to n[j, i..i+3]
  movups [rcx + rax], xmm0

  ; End of procedure
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; void simd_matrix_product(uint rows1,      // rdi ;;
;;                          uint cols1,      // rsi ;;
;;                          uint cols2,      // rdx ;;
;;                          const float* m,  // rcx ;;
;;                          const float* nt, // r8  ;;
;;                          float* p)        // r9  ;;
;;                                                  ;;
;; Computes the product between matrices m and n,   ;;
;; and stores the results in p.                     ;;
;;                                                  ;;
;; Assumes:                                         ;;
;;  - rows1 >= 4, cols1 >= 4 and cols2 >= 4.        ;;
;;  - rows2 = cols1 (note that rows2 is not         ;;
;;    provided as an argument).                     ;;
;;  - matrix nt is the transpose of n.              ;;
;;  - cols2 corresponds to the number of columns in ;;
;;    the original matrix n *before* transposition. ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

simd_matrix_product:
  push rbp
  mov rbp, rsp
  push rbx
  push r12
  push r13
  push r14
  push r15

  ; Clear high 32 bits of rdi, rsi and rdx
  mov edi, edi
  mov esi, esi
  mov edx, edx

  ; Row index (i) in the resulting matrix
  xor r12, r12

  ; Iterate over rows in the resulting matrix
product_rows_loop:

  ; Column index (j) in the resulting matrix
  xor r13, r13

  ; Iterate over columns in the resulting matrix
product_cols_loop:

  ; Compute value for cell (i,j) in the result matrix and store it in xmm2
  call product_compute_ij

  ; Save value in xmm2 in cell (i,j) in the result matrix
  call product_save_ij

  ; Increase col index and iterate
  inc r13
  cmp r13, rdx
  jne product_cols_loop

  ; Increase row index and iterate
  inc r12
  cmp r12, rdi
  jne product_rows_loop

  pop r15
  pop r14
  pop r13
  pop r12
  pop rbx
  pop rbp
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_compute_ij computes the value for cell (i,j) in      ;;
;; the resulting matrix. Expects rdi = rows, rsi=cols1, rcx=pointer to m, ;;
;; r8=pointer to nt (transpose of n), r9=pointer to resulting matrix,     ;;
;; r12=i, r13=j.                                                          ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_compute_ij:

  ; Initialize accumulator
  pxor xmm2, xmm2

  ; Internal loop index (k)
  xor r14, r14

product_internal_loop:

  ; Load m[i, k..k+3] into xmm0 and n[k..k+3, j] into xmm1
  call product_ijk_fetch

  ; Compute dot-product between xmm0 and xmm1 and store it in xmm0
  call product_ijk_reduce

  ; Accumulate results
  addss xmm2, xmm0

  ; Compute number of remaining cells left
  mov r15, rsi             ; r14 = cols1
  sub r15, r14             ; r14 = cols1 - current cell
  sub r15, 4               ; r14 = cols1 - current cell - 4

  ; Case 1: no cells left
  jz exit_product_internal_loop

  ; Case 2: 4 or more cells left
  cmp r15, 4
  jge product_internal_loop_next

  ; Case 3: 1, 2 or 3 columns left
  mov r14, rsi             ; k = r14 = cols1
  sub r14, 4               ; k = r14 = cols1 - 4

  call product_ijk_fetch ; Load m[i, cols1-4..cols1] into xmm0
                         ; and  n[cols1-4..cols1, j] into xmm1

  ; 1 column left
  cmp r15, 1
  je product_internal_loop_1_column_left

  ; 2 columns left
  cmp r15, 2
  je product_internal_loop_2_columns_left

  ; 3 columns left
  jmp product_internal_loop_3_columns_left

  ; Case 3: 1 column left
product_internal_loop_1_column_left:

  ; Compute the product between the last float in
  ; xmm0 and xmm1 and store it in xmm0
  call product_ijk_reduce_1_column_left

  ; Finish reduction
  jmp product_internal_loop_special_case_finish_reduction

  ; Case 3: 2 columns left
product_internal_loop_2_columns_left:

  ; Compute dot-product between the last 2 floats in
  ; xmm0 and xmm1 and store it in xmm0
  call product_ijk_reduce_2_columns_left

  ; Finish reduction
  jmp product_internal_loop_special_case_finish_reduction

  ; Case 3: 3 columns left
product_internal_loop_3_columns_left:

  ; Compute dot-product between the last 3 floats in
  ; xmm0 and xmm1 and store it in xmm0
  call product_ijk_reduce_3_columns_left

product_internal_loop_special_case_finish_reduction:

  ; Accumulate results
  addss xmm2, xmm0

  jmp exit_product_internal_loop

product_internal_loop_next:
  add r14, 4               ; Advance 4 cells
  jmp product_internal_loop

exit_product_internal_loop:
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_ijk_fetch loads m[i, cols1-4..cols1] into xmm0 ;;
;; and n[cols1-4..cols1, j] into xmm1. Expects r12=i, r13=j, r14=k, ;;
;; rsi=cols1, rcx=pointer to m, r8=pointer to nt (transpose of n).  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_ijk_fetch:

  ; Compute offset for m(i, k)
  mov rax, r12              ; rax = i
  push rdx
  mul rsi                   ; rax = i * cols1
  add rax, r14              ; rax = (i * cols1) + k
  mov rbx, 4                ; rbx = 4
  mul rbx                   ; rax = 4 * ((i * cols1) + k)
  pop rdx

  ; Store m(i, k..k+3) in xmm0
  movups xmm0, [rcx + rax]  ; Copy values at current offset in matrix m

  ; Compute offset for n^t(j, k)
  mov rax, r13              ; rax = j
  push rdx
  mul rsi                   ; rax = j * cols1
  add rax, r14              ; rax = (j * cols1) + k
  mov rbx, 4                ; rbx = 4
  mul rbx                   ; rax = 4 * ((j * cols1) + k)
  pop rdx

  ; Store nt(j, k..k+3) in xmm1
  movups xmm1, [r8 + rax]  ; Copy values at current offset in matrix nt

  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_ijk_reduce computes the dot product between ;;
;; xmm0 and xmm1 and stores it in the low dword in xmm0.         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_ijk_reduce:
  dpps xmm0, xmm1, 0xF1
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_ijk_reduce computes the dot product between ;;
;; the high dword in  xmm0 and xmm1 and stores it in the low     ;;
;; dword in xmm0.                                                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_ijk_reduce_1_column_left:
  dpps xmm0, xmm1, 0x81
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_ijk_reduce computes the dot product between ;;
;; the two higher dwords in  xmm0 and xmm1 and stores it in the  ;;
;; low dword in xmm0.                                            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_ijk_reduce_2_columns_left:
  dpps xmm0, xmm1, 0xC1
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_ijk_reduce computes the dot product between ;;
;; the three higher dwords in  xmm0 and xmm1 and stores it in    ;;
;; the low dword in xmm0.                                        ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_ijk_reduce_3_columns_left:
  dpps xmm0, xmm1, 0xE1
  ret

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Procedure product_save_ij stores the computed value for (i,j) ;;
;; in the resulting matrix. Expects r12=i, r13=j, rdx=cols2,     ;;
;; r9=pointer to the resulting matrix.                           ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

product_save_ij:

  ; Compute offset for p(i, j)
  mov rax, r12              ; rax = i
  push rdx
  mul rdx                   ; rax = i * cols2
  add rax, r13              ; rax = (i * cols2) + j
  mov rbx, 4                ; rbx = 4
  mul rbx                   ; rax = 4 * ((i * cols2) + j)
  pop rdx

  ; Store xmm0 in p(i, j)
  movss [r9 + rax], xmm2    ; Copy values at current offset in matrix m

  ret
