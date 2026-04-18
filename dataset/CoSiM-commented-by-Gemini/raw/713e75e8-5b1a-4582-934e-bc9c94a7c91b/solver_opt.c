
/**
 * @file solver_opt.c
 * @brief An optimized, modular implementation of a matrix solver.
 *
 * This file provides a version of the matrix solver that is both modular
 * (using helper functions for each operation) and optimized at a low level.
 * It applies classic C optimization techniques like using the `register`
 * keyword and manual pointer arithmetic to improve performance.
 */
#include "utils.h"


/**
 * @brief Computes the transpose of a square matrix using pointer arithmetic.
 * @param N The dimension of the matrix.
 * @param a The input matrix.
 * @return A new matrix containing the transpose of a.
 */
double* transpose(int N, double *a){
	double* transpose = (double*)malloc((N * N) * sizeof(double));
	register int i, j;
  		
    for (i = 0; i < N; i++){
		register double *pa = &a[i * N];
    	register double *tb = &transpose[i];

  		for (j = 0; j < N; j++){
  			*tb = *pa;
  			tb += N; // Move to the next row in the same column of the transpose matrix.
  			pa++;    // Move to the next element in the row of the original matrix.
  		}
	}

  	return transpose;
}

/**
 * @brief Performs an in-place addition of two matrices (A = A + B) using pointer arithmetic.
 * @param N The dimension of the matrices.
 * @param A The first matrix, which is modified in-place.
 * @param B The second matrix.
 * @return A pointer to the modified matrix A.
 */
double* add(int N, double *A, double *B){
	register int i, j;

	for(i = 0 ; i < N ; i++){
		register double *pa = &A[i * N];
		register double *pb = &B[i * N];
		for(j = 0; j < N ; j++){
			*pa += *pb;
			pa++;
			pb++;
		}
	}
	return A;
}


/**
 * @brief Computes the product of a matrix with its transpose (a * a^T).
 * @param N The dimension of the matrix.
 * @param a The input matrix.
 * @return A new matrix containing the result of a * a^T.
 */
double* multiply(int N, double *a) {
	double* c = (double*)calloc(sizeof(double), (N * N));
	register int i, j, k;

	for (i = 0 ; i < N ; i++){
		register double *orig_pa = &a[i * N];
		for (j = 0 ; j < N ; j++){
			register double *pa = orig_pa;
			register double *pb = &a[j * N];
			register double suma = 0;
			// This inner loop computes the dot product of row i and row j of 'a'.
	    	for (k = 0 ; k < N ; k++){
	    		suma += *pa * *pb;
	    		pa++;
	    		pb++;
	      	}
	      	c[i * N + j] = suma;
   		}
	}

	return c;
}

// Utility function to find the minimum of two integers.
int min(int a, int b){
	return (a < b) ? a : b;
}

/**
 * @brief Performs a non-standard triangular matrix multiplication, optimized with pointers.
 * @param N The dimension of the matrices.
 * @param a The first input matrix.
 * @param b The second input matrix.
 * @return A new matrix containing the result.
 */
double* multiply_two_triunghiular(int N, double *a, double* b) {
	double* c = (double*)malloc((N * N) * sizeof(double));
	register int i, j, k;

	for (i = 0 ; i < N ; i++){
		register double *orig_pa = &a[i * N];
		for (j = 0 ; j < N ; j++){
			register double *pa = orig_pa;
			register double *pb = &b[j];
			register double suma = 0;
	    	for (k = 0 ; k < min(i + 1, j + 1) ; k++){
	    		suma += *pa * *pb;
	    		pa++;
	    		pb += N;
	      	}
	      	c[i * N + j] = suma;
   		}
	}

	return c;
}

/**
 * @brief Multiplies an upper triangular matrix with a general matrix (c = a * b).
 * @param N The dimension of the matrices.
 * @param a The upper triangular input matrix.
 * @param b The general input matrix.
 * @return A new matrix containing the result.
 */
double* multiply_upper_triunghiular(int N, double *a, double* b) {
	double* c = (double*)malloc((N * N) * sizeof(double));
	register int i, j, k;

	for (i = 0 ; i < N ; i++){
		register double *orig_pa = &a[i * N + i];
		for (j = 0 ; j < N ; j++){
			register double *pa = orig_pa;
    		register double *pb = &b[j + N * i];
			register double suma = 0;
			// Inner loop starts from k=i, reflecting that 'a' is upper triangular.
	    	for (k = i ; k < N ; k++){
	    		suma += *pa * *pb;
      			pa++;
      			pb += N;
	      	}
	      	c[i * N + j] = suma;
   		}
	}

	return c;
}


/**
 * @brief Performs a sequence of matrix operations using optimized helper functions.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix.
 *
 * @note This function computes the expression: C = A * (B * B^T) + A^T * A.
 *       It uses specialized and micro-optimized helper functions for each step.
 */
double* my_solver(int N, double *A, double* B) {
	// Step 1: Compute A-transpose.
	double* a_transpose = transpose(N, A);

	// Step 2: Compute b_bt = B * B^T using a specialized multiply function.
	double* b_bt = multiply(N, B);

	// Step 3: Compute a_b_bt = A * (B * B^T) using an upper-triangular multiply.
	double* a_b_bt = multiply_upper_triunghiular(N, A, b_bt);

	// Step 4: Compute at_a = A^T * A using a custom triangular multiply.
	double* at_a = multiply_two_triunghiular(N, a_transpose, A);

	// Step 5: Perform the final addition.
	double* c = add(N, a_b_bt, at_a);

	// Cleanup intermediate allocations.
	free(a_transpose);
	free(b_bt);
	free(at_a);

	return c;
}
