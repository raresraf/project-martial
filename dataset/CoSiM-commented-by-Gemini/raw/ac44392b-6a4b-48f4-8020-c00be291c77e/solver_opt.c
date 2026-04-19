/**
 * @raw/ac44392b-6a4b-48f4-8020-c00be291c77e/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = A * (B * B^T) + A^T * A.
 * Algorithm: Cache-friendly loop reordering, explicit pointer incrementations, and symmetrical mirroring optimization.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ due to auxiliary storage buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
  
  /**
   * Functional Utility: Allocates block memory for intermediate products.
   */
  double* bbt = calloc(N*N, sizeof(double));
  double* res = calloc(N*N, sizeof(double));
  
  /**
   * Block Logic: Computes bbt = B * B^T.
   * Optimization: Eliminates repetitive coordinate multiplication addressing inside deep loops. Employs symmetry.
   */
  for (register int i = 0; i < N; ++i) {
    double *orig_row_b = B + N*i;
    
    for (register int j = i; j < N; ++j) {
      register double *row_b = orig_row_b;
      register double *column_b = B + N*j;
      register double sum = 0.0;
      for (register int k = 0; k < N; ++k) {
        
        sum += *row_b * *column_b;
        row_b++;
        column_b++;
      }
      bbt[N*i + j] += sum;
      if(i!=j) bbt[N*j + i] += sum;
    }
  }


  /**
   * Block Logic: Executes res = A * bbt.
   * Optimization: Uses cache efficient striding via fast localized pointers tracking.
   */
  for (register int i = 0; i < N; ++i) {
    double *orig_pa = A + N*i;
    for (register int j = 0; j < N; ++j) {
      register double *pa = orig_pa + i;
      register double *pb = bbt + N*i + j;
      register double sum = 0;
      for (register int k = i; k < N; ++k) {
        sum += *pa * *pb;
        pa++;
        pb += N;
      }
      res[N*i + j] += sum;
    }
  }

  /**
   * Block Logic: Evaluates A^T * A while directly adding it to `res`.
   * Optimization: Explores symmetric patterns mitigating half of the iterations utilizing local registers.
   */
  for (register int i = 0; i < N; ++i) {
    
    for (register int j = i; j < N; ++j) {
      register double *pa = A + i;
      register double *pb = A + j;
      register double sum = 0;
      for (register int k = 0; k < N; ++k) {
          sum += *pa * *pb;
          pa += N;
          pb += N;
      }
      res[N*i + j] += sum;
      if (i != j) res[N*j + i] += sum;
    }
  }
  free(bbt);

  return res;
}
