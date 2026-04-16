/**
 * @file solver_opt.c
 * @brief An optimized C implementation of a matrix solver.
 *
 * This file provides a `my_solver` implementation that computes the expression
 * `C = (A * B) * B^T + A^T * A` within a single function. It employs several
 * manual C-level optimizations, including heavy use of pointer arithmetic and the
 * `register` keyword, to improve performance.
 */
#include "utils.h"


/**
 * @brief Computes a matrix expression using manually optimized C loops.
 *
 * This function calculates the result of `C = (A * B) * B^T + A^T * A`. The
 * entire computation is performed inside a single function, using three
 * sequential loop blocks. The implementation is heavily optimized with
 * low-level C techniques.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A. Assumed to be upper triangular.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the final result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The optimization techniques include:
 * - **Register Hinting:** All loop counters and frequently accessed pointers are
 *   declared with the `register` keyword to suggest storage in CPU registers.
 * - **Pointer Arithmetic:** Array indexing is completely replaced by direct pointer
 *   manipulation for traversing matrices, which can reduce address calculation overhead.
 * - **Combined Final Loop:** The final step, which computes `AB * B^T`, also adds the
 *   pre-computed `A^T * A` result in the same loop, improving data locality and
 *   reducing the need for a separate summation loop.
 */
double* my_solver(int N, double *A, double* B) {

	printf("OPT SOLVER
");
	double *AtA = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));
	double *res = calloc(N * N, sizeof(double));
	
	
	for (register int i = 0; i < N; ++i) {
		
		register double *AtA_ptr 	= AtA + i * N;
		register double *At_ptr 	= A + i;
		
		for (register int j = 0; j < N; ++j) {
			register double *copy_At_ptr = At_ptr;
			
			register double *A_ptr = A + j;
			register double sum = 0;
			
			
			for (register int k = 0; k <= j && k <= i; ++k) { 
				sum += (*copy_At_ptr) * (*A_ptr);
				copy_At_ptr += N;
				A_ptr += N;
			}
			*AtA_ptr = sum;
			AtA_ptr++;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *AB_ptr 	= AB + i * N;
		register double *A_ptr 		= A + i * N + i; 

		for (register int j = 0; j < N; ++j) {
			register double *copy_A_ptr = A_ptr;
			register double *B_ptr = B + i * N + j; 
			register double sum = 0;

			for (register int k = i; k < N; ++k) {
				sum += (*copy_A_ptr) * (*B_ptr);
				copy_A_ptr++;
				B_ptr += N;
			}
			*AB_ptr = sum;
			AB_ptr++;
		}
	}

	
	for (register int i = 0; i < N; ++i) {
		register double *AtA_ptr 	= AtA + i * N;
		register double *AB_ptr 	= AB + i * N;
		register double *res_ptr 	= res + i * N;

		for (register int j = 0; j < N; ++j) {
			register double *copy_AB_ptr = AB_ptr;
			register double *Bt_ptr = B + j * N;
			register double sum = 0;

			for (register int k = 0; k < N; ++k) {
				sum += (*copy_AB_ptr) * (*Bt_ptr);
				copy_AB_ptr++;
				Bt_ptr++;
			}
			*res_ptr = sum + *AtA_ptr;
			AtA_ptr++;
			res_ptr++;
		}
	}

	free(AtA);
	free(AB);

	return res;	
}
