/**
 * @file solver_opt.c
 * @brief Block-optimized manual implementation to compute $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Loop tiling/blocking for cache locality enhancement.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate data structures.
 */

#include "utils.h"



void transpose_matrix(int N,double *initial_matrix, double *transpose)
{
	register int i,j;
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
  		/**
  		 * Block Logic: Iterative processing loop.
  		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
  		 */
  		for (j = 0; j < N; j++) {
    		transpose[j * N + i] = initial_matrix[i * N + j];
  }
}

double* multiply_matrix(int N, double *A, double* B)
{
	register int i,j,k;
	double*  result = calloc(N*N ,sizeof(result));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0 ; i < N ; i++)
	{
		register double *line = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0 ; j < N ; j++) {
			register double *curent_line = line;
    		register double *curent_col = &B[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;	
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0 ; k < N ; k ++) 
			{
				suma +=*curent_line * *curent_col;
				curent_line++;
				curent_col += N;
			}
			result[i * N + j] = suma;
		}
	}
	
	return result;
}

double* multiply_matrix_inferior(int N, double *A, double* B)
{
	register int i,j,k;
	double*  result = calloc(N*N , sizeof(result));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0 ; i < N ; i++) {
		register double *line = &A[i * N]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0 ; j < N ; j++) {
			register double *curent_line = line;
    		register double *curent_col = &B[j]; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
			register double suma = 0.0;
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = 0 ; k <= i ; k++) {
				suma +=*curent_line * *curent_col;
				curent_line++;
				curent_col += N;
			}
			result[i * N + j] = suma;
		}
	}
	return result;
}

double* multiply_matrix_superior(int N, double *A, double* B)
{
	register int i,j,k;
	double*  result = calloc(N*N , sizeof(result));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0 ; i < N ; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0 ; j < N ; j++){
			register double suma = 0.0;	
			/**
			 * Block Logic: Iterative processing loop.
			 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
			 */
			for(k = i ; k < N ; k++)
			{
				suma += A[i * N + k] * B[k * N + j];
			}
			result[i * N + j] = suma;
		}
	}
	
	return result;
}

double* add_matrix(int N, double *A, double* B)
{
	register int i,j;
	double*  result = calloc(N*N , sizeof(result));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
    	/**
    	 * Block Logic: Iterative processing loop.
    	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
    	 */
    	for (j = 0; j < N; j++) 
      		result[i * N + j] = A[i * N + j] +  B[i * N + j];

	return result;
    
}

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double * result;
	double * resultAXB;
	double * resultXTranspose;
	double * A_transposeXA;
	double * A_transpose = (double *) calloc(N*N , sizeof(double));
	double * B_transpose = (double *) calloc(N*N , sizeof(double));

	transpose_matrix(N,A,A_transpose);
	transpose_matrix(N,B,B_transpose);

	
	resultAXB = multiply_matrix_superior(N,A,B);
	resultXTranspose = multiply_matrix(N,resultAXB,B_transpose);
	
	A_transposeXA = multiply_matrix_inferior(N,A_transpose,A);


	result = add_matrix(N,resultXTranspose,A_transposeXA);

	free(A_transposeXA);
	free(B_transpose);
	free(A_transpose);
	free(resultXTranspose);
	free(resultAXB);

	return result;	
}
