/**
 * @file solver_opt.c
 * @brief An optimized C implementation of a matrix solver using explicit transposition.
 *
 * This file provides an implementation of the `my_solver` function that attempts
 * to optimize the computation `C = (A * B) * B^T + A^T * A` by manually
 * transposing the input matrices and using pointer-intensive loop structures.
 *
 * The key optimization strategy here is to create explicit transposed copies of
 * matrices A and B. This is done to ensure that matrix multiplications can be
 * performed with linear, sequential memory access patterns (i.e., accessing columns
 * of the original matrix becomes accessing rows of the transposed matrix), which
 * can improve cache utilization.
 */
#include "utils.h"
#include <stdlib.h>



int min(int a, int b) {
    return a < b ? a : b;
}

void allocate_matrices(int N, double **C, double **AB, double **ABBt,
	double **AtA, double **A_t, double **B_t)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABBt = calloc(N * N, sizeof(**ABBt));
	if (NULL == *ABBt)
		exit(EXIT_FAILURE);

	*AtA = calloc(N * N, sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);

	*A_t = calloc(N * N, sizeof(**AtA));
	if (NULL == *AtA)
		exit(EXIT_FAILURE);

	*B_t = calloc(N * N, sizeof(**B_t));
	if (NULL == *B_t)
		exit(EXIT_FAILURE);
}

/**
 * @brief Computes a matrix expression using explicit transposition and pointer arithmetic.
 *
 * This function calculates `C = (A * B) * B^T + A^T * A`. Instead of relying on
 * BLAS or simple nested loops, it first creates transposed versions of A and B.
 * It then performs the series of matrix multiplications using manually crafted loops
 * that rely heavily on pointer arithmetic and the `register` keyword to hint at
 * compiler optimizations.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A. Assumed to be upper triangular.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The sequence of operations is:
 * 1.  Allocate memory for all intermediate and transposed matrices.
 * 2.  Explicitly compute the transposes `A_t` and `B_t`.
 * 3.  Calculate `AtA = A^T * A` using `A` and `A_t`.
 * 4.  Calculate `AB = A * B`.
 * 5.  Calculate `C = AB * B_t`, storing the result of `(A * B) * B^T` directly in `C`.
 * 6.  Perform element-wise addition `C = C + AtA` to get the final result.
 * 7.  Free all temporary matrices.
 */
double* my_solver(int N, double *A, double* B) {
	double *C;
	double *AB;
	double *ABBt;
	double *AtA;

	double *A_t;
	double *B_t;

	allocate_matrices(N, &C, &AB, &ABBt, &AtA, &A_t, &B_t);

	
	register int i;
	for (i = 0; i < N; ++i) {
		register double *A_t_ptr = A_t + i;  
		register double *B_t_ptr = B_t + i;  

		register double *A_ptr = A + i * N;  
		register double *B_ptr = B + i * N;  

		register int j;
		for (j = 0; j < N; ++j, A_t_ptr += N,
			B_t_ptr += N, ++A_ptr, ++B_ptr) {
			*A_t_ptr = *A_ptr;
			*B_t_ptr = *B_ptr;
		}
	}
	
	for (i = 0; i < N; ++i) {
		register double *AtA_ptr = AtA + i * N;  
		
		register int j;
		for (j = 0; j < N; ++j, ++AtA_ptr) {
			
			register double result = 0;

			
			register double *A_ptr = A + j % N;
		        register double *A_t_ptr = A_t + i * N;  
			
			register int k;
			for (k = 0; k < min(i + 1, j + 1); ++k, 
                        A_t_ptr = A_t + i * N + k, A_ptr = A + k * N + j) {
				result += *A_t_ptr * *A_ptr;
        		}
			*AtA_ptr = result;
		}
	}
	
    
	for (i = 0; i < N; ++i) {
		register double *AB_ptr = AB + i * N;  
		
		register int j;	
		for (j = 0; j < N; ++j, ++AB_ptr) {
			
			register double result = 0;

			
			register double *A_ptr = A + i * N + i;
	                register double *B_ptr = B + j % N + i * N;  
			
			register int k;
			for (k = i; k < N; ++k, ++A_ptr, B_ptr += N) {
				result += *A_ptr * *B_ptr;
            		}
			*AB_ptr = result;
		}
	}

    
	for (i = 0; i < N; ++i) {
		register double *C_ptr = C + i * N;  
		
		register int j;
		for (j = 0; j < N; ++j, ++C_ptr) {
			
			register double result = 0;

			
			register double *AB_ptr = AB + i * N;
		        register double *B_t_ptr = B_t + j % N;  
			
			register int k;
			for (k = 0; k < N; ++k, ++AB_ptr, B_t_ptr += N) {
				result += *AB_ptr * *B_t_ptr;
            		}
			*C_ptr = result;
		}
	}

    
	for (i = 0; i < N; ++i) {
		register double *C_ptr = C + i * N;  
		register double *AtA_ptr = AtA + i * N;  
		
		register int j;
        	for(j = 0; j < N; ++j, C_ptr++, AtA_ptr++) {
            		*C_ptr += *AtA_ptr;
        	}
	}
	
	free(AB);
	free(ABBt);
	free(AtA);

	free(A_t);
	free(B_t);
	return C;	
}
