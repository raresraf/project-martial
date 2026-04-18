
/**
 * @file solver_opt.c
 * @brief An optimized, loop-based implementation of a matrix solver.
 *
 * This file contains a version of the non-optimized solver that has been
 * manually optimized using low-level C techniques. The algorithmic steps
 * are the same, but the implementation within the loops is changed to
 * improve performance.
 */
#include "utils.h"

/**
 * @brief Performs a sequence of matrix operations using optimized naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * @note This function computes the expression: C = (A * B) * B^T + A^T * A.
 *       It uses the same step-by-step logic as the non-optimized version but
 *       applies the following micro-optimizations within the loops:
 *       1.  **Register Keyword**: Hints to the compiler to store loop counters and
 *           pointers in CPU registers for faster access.
 *       2.  **Pointer Arithmetic**: Manually calculates pointers to rows/columns
 *           outside of inner loops and uses simple pointer increments inside,
 *           aiming to reduce address calculation overhead.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER
");
	register int i, j, k;
	register double  *pa, *pb, *orig_pa;
	register double suma;
	// Allocate memory for intermediate matrices.
	double *M1 = (double *)calloc(N * N, sizeof(double)); 
	double *M2 = (double *)calloc(N * N, sizeof(double)); 
	double *M3 = (double *)calloc(N * N, sizeof(double)); 
	if (!M1 || !M2 || !M3)
		return NULL;

	// Block Logic: Step 1 - Compute M1 = A * B, treating A as an upper triangular matrix.
	for (i = 0; i < N; i++) {
		orig_pa = &(A[i * N + i]);
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &(B[i * N + j]);
			suma = 0.0;
			for(k = i; k < N; k++) {
				suma += *(pa++) * *pb;
				pb += N;
			}
			M1[i * N + j] = suma;
		}
	}

	// Block Logic: Step 2 - Compute M2 = M1 * B^T, which is (A * B) * B^T.
	for (i = 0; i < N; i++) {
		orig_pa = &(M1[i * N]);
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &(B[j * N]);
			suma = 0.0;
			for(k = 0; k < N; k++) {
			 	suma += *pa * *pb;
				pa++;
				pb++;
			}
			M2[i * N + j] = suma;
		}
	}

	// Block Logic: Step 3 - Compute M3 = A^T * A.
	register int end;
	for (i = 0; i < N; i++) {
		orig_pa = &(A[i]);
		for (j = 0; j < N; j++) {
			if (i < j)
				end = i;
			else
				end = j;
			pa = orig_pa;
			pb = &(A[j]);
			suma = 0.0;
			for(k = 0; k <= end; k++) {
			 	suma += *pa * *pb;
				pa += N;
				pb += N;
			}
			M3[i * N + j] = suma;
		}
	}

	// Block Logic: Step 4 - Perform the final addition M2 = M2 + M3.
	// This loop iterates down the columns for better memory access patterns.
	for (i = 0; i < N; i++) {
		pa = &(M2[i]);
		pb = &(M3[i]);
		for (j = 0; j < N; j++) {
			*pa += *pb;
			pa += N;
			pb += N;
		}
	}
	
	// Free intermediate matrices. The result is in M2.
	free(M1);
	free(M3);
	return M2;
}
