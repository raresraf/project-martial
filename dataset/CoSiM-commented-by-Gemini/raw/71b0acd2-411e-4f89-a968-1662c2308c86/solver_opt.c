
/**
 * @file solver_opt.c
 * @brief An optimized C implementation of a matrix solver using loop-level optimizations.
 *
 * This file provides a version of the matrix solver that applies low-level
 * optimizations like the `register` keyword and manual pointer arithmetic.
 * It reuses memory and exploits the symmetry of intermediate calculations.
 */
#include "utils.h"


/**
 * @brief Allocates memory for several intermediate and final matrices.
 * @param N The dimension of the matrices.
 * @param C Pointer to the final result matrix.
 * @param BB_t Pointer to an intermediate matrix for B*B^T.
 * @param ABB_t Pointer to an intermediate matrix (allocated but unused).
 */
void allocate_matrices(int N, double **C, double **BB_t, double **ABB_t)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*BB_t = malloc(N * N * sizeof(**BB_t));
	if (NULL == *BB_t)
		exit(EXIT_FAILURE);

	// Note: ABB_t is allocated here but is not used in the my_solver function.
	*ABB_t = malloc(N * N * sizeof(**ABB_t));
	if (NULL == *ABB_t)
		exit(EXIT_FAILURE);

}

/**
 * @brief Performs a sequence of matrix operations using optimized naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C.
 *
 * @note This function computes the expression: C = A * (B * B^T) + A^T * A.
 *       It uses manual pointer arithmetic and the `register` keyword as optimization hints.
 *       It also reuses the memory for the final matrix `C` to store an intermediate result.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER
");

	double *C;
	double *BB_t;
	double *ABB_t;
	register int i, j, k;

	
	allocate_matrices(N, &C, &BB_t, &ABB_t);

	
	// Block Logic: Compute BB_t = B * B^T.
	// Optimization: Since the result is symmetric, only the lower triangle (and diagonal)
	// is computed, and the upper triangle is filled by copying.
	for (i = 0; i < N; ++i) {
		double *orig_pb = B + (i * N);
		double *orig_pbt = BB_t + (i * N);
		for (j = 0; j <= i; ++j) {
			register double *pb = orig_pb;
			register double *pb_t = B + (j * N);
			register double suma = 0.0;
			// Computes the dot product of row i and row j of B.
			for (k = 0; k < N; ++k) {
				suma += (*pb) * (*pb_t);
				pb++;
				pb_t++;
			}
			*orig_pbt = suma;
			orig_pbt++;

			// Exploit symmetry.
			if(i != j)
				BB_t[j * N + i] = suma;
		}
	}


	
	// Block Logic: Compute C = A^T * A.
	// Optimization: Reuses the final result matrix C to store this intermediate result.
	// Also exploits the symmetry of A^T * A.
	for (i = 0; i < N; ++i) {
		for (j = i; j < N; ++j) {
			register double suma = 0.0;
			register double *a = A;
			// Computes the dot product of column i and column j of A.
			for (k = 0; k <= i; ++k) {
				suma += *(a + i) * 
								(*(a + j));
				a += N;
			}
			C[i * N + j] = suma;
			C[j * N + i] = suma;
		}
	}
	
	// Block Logic: Compute C = C + A * BB_t, where A is treated as upper triangular.
	// This adds the second term of the main expression to the first term already in C.
	for (i = 0; i < N; ++i) {
		double *orig_pa = A + (i * N + i);
		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			int a = i * N + j;
			register double *pb = BB_t + a;
			register double suma = 0.0;
			// Inner loop for k starts from i, assuming A is upper triangular.
			for (k = i; k < N; ++k) {
				suma += (*pa) * (*pb);
				pa++;
				pb += N;
			}
			C[a] += suma;
		}
	}

	
	// Free the memory used for intermediate matrices.
	free(ABB_t); // This was allocated but not used.
	free(BB_t);

	return C;
}
