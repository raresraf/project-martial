/**
 * @file solver_neopt.c
 * @brief This file implements a non-optimized matrix solver, featuring basic matrix operations
 * such as addition, and specialized multiplications for normal-transpose and upper-triangular matrices.
 * This implementation serves as a baseline for performance comparisons, contrasting with optimized
 * versions like those utilizing BLAS.
 *
 * Algorithm: Naive matrix operations (addition, multiplication), no advanced optimizations (e.g., tiling, Strassen).
 * Time Complexity: Predominantly O(N^3) for matrix multiplications, and O(N^2) for addition, where N is matrix dimension.
 * Space Complexity: O(N^2) for storing intermediate matrices.
 */

#include "utils.h"




/**
 * @brief Performs element-wise addition of two matrices.
 * This function computes C = A + B for N x N matrices stored in row-major order.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the first input matrix A.
 * @param b Pointer to the second input matrix B.
 * @param c Pointer to the output matrix C, where the result will be stored.
 */
void add(int N, double *a, double *b, double *c) {
	int i;
	// Block Logic: Iterate through all elements of the matrices.
	// Precondition: Matrices a, b, and c are allocated for N*N doubles.
	// Invariant: After each iteration, c[i] contains the sum of a[i] and b[i].
	for (i = 0; i < N * N; i++) {
		c[i] = a[i] + b[i];
	}
}



/**
 * @file solver_neopt.c
 * @brief This file implements a non-optimized matrix solver, featuring basic matrix operations
 * such as addition, and specialized multiplications for normal-transpose and upper-triangular matrices.
 * This implementation serves as a baseline for performance comparisons, contrasting with optimized
 * versions like those utilizing BLAS.
 *
 * Algorithm: Naive matrix operations (addition, multiplication), no advanced optimizations (e.g., tiling, Strassen).
 * Time Complexity: Predominantly O(N^3) for matrix multiplications, and O(N^2) for addition, where N is matrix dimension.
 * Space Complexity: O(N^2) for storing intermediate matrices.
 */

#include "utils.h"




/**
 * @brief Performs element-wise addition of two matrices.
 * This function computes C = A + B for N x N matrices stored in row-major order.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the first input matrix A.
 * @param b Pointer to the second input matrix B.
 * @param c Pointer to the output matrix C, where the result will be stored.
 */
void add(int N, double *a, double *b, double *c) {
	int i;
	// Block Logic: Iterate through all elements of the matrices.
	// Precondition: Matrices a, b, and c are allocated for N*N doubles.
	// Invariant: After each iteration, c[i] contains the sum of a[i] and b[i].
	for (i = 0; i < N * N; i++) {
		c[i] = a[i] + b[i];
	}
}


/**
 * @brief Computes the product of a normal matrix with the transpose of another normal matrix.
 * Specifically, this function calculates a portion of C = A * A_transpose, storing the result
 * in a symmetric manner due to the property of the operation (C[i][j] = C[j][i]).
 * This function is optimized for symmetry by calculating only the upper triangle and copying to the lower.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the input matrix A (N x N).
 * @param c Pointer to the output matrix C (N x N), which will store A * A_transpose.
 */
void normal_x_normal_transpose(int N, double *a, double *c) {

	int i, j, k;

	// Block Logic: Iterate over the upper triangle and diagonal of the resultant matrix C.
	// Due to symmetry (C[i][j] = C[j][i]), only these elements need to be explicitly calculated.
	// Precondition: `a` and `c` are initialized N x N matrices.
	// Invariant: At the end of the outer loops, the upper triangular and diagonal elements of C are computed.
	for (i = 0; i < N; i++) {
		
		for (j = 0; j <= i; j++) {
			// Initialize current element of C to 0 before summation.
			c[i * N + j] = 0.0;
			// Block Logic: Perform the dot product for element C[i][j].
			// The inner loop iterates through the columns of A and rows of A_transpose.
			// Invariant: `c[i * N + j]` accumulates the sum of products for the (i, j) element.
			for (k = 0; k < N; k++) {
				
				c[i * N + j] += a[i * N + k] * a[j * N + k];
				
				// Functional Utility: Exploit symmetry (C[j][i] = C[i][j]) to avoid redundant calculations.
				// This directly assigns the computed value to its symmetric counterpart.
				c[j * N + i] = c[i * N + j];
			}
		}
	}
}


/**
 * @brief Performs matrix multiplication where the first matrix is upper triangular.
 * This function computes C = A * B, where A is implicitly upper triangular and both B and C are normal matrices.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the input matrix A (N x N), implicitly upper triangular.
 * @param b Pointer to the input matrix B (N x N).
 * @param c Pointer to the output matrix C (N x N), which will store A * B.
 */
void upper_x_normal(int N, double *a, double *b, double *c) {

	int i, j, k;

	// Block Logic: Iterate through each element of the resulting matrix C.
	// Precondition: `a`, `b`, and `c` are initialized N x N matrices.
	// Invariant: Each element `c[i * N + j]` will store the correct sum of products.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			// Initialize current element of C to 0 before summation.
			c[i * N + j] = 0.0;
			// Block Logic: Perform the dot product for element C[i][j], considering A is upper triangular.
			// The inner loop starts from `k = i` because elements `a[i * N + k]` where `k < i` are zero
			// in an upper triangular matrix, thus skipping unnecessary multiplications.
			for (k = i; k < N; k++) {
				c[i * N + j] += a[i * N + k] * b[k * N + j];
			}
		}
	}
}

/**
 * @brief Computes the product of an upper triangular matrix's transpose with another upper triangular matrix.
 * Specifically, this function calculates a portion of C = A_transpose * A, where A is an upper triangular matrix.
 * The result is symmetric, and only the upper triangle and diagonal are explicitly computed.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the input matrix A (N x N), implicitly upper triangular.
 * @param c Pointer to the output matrix C (N x N), which will store A_transpose * A.
 */
void upper_transpose_x_upper(int N, double *a, double *c) {

	int i, j, k;

	// Block Logic: Iterate over the upper triangle and diagonal of the resultant matrix C.
	// Due to symmetry (C[i][j] = C[j][i]), only these elements need to be explicitly calculated.
	// Precondition: `a` and `c` are initialized N x N matrices.
	// Invariant: At the end of the outer loops, the upper triangular and diagonal elements of C are computed.
	for (i = 0; i < N; i++) {
		
		for (j = 0; j <= i; j++) {
			// Initialize current element of C to 0 before summation.
			c[i * N + j] = 0.0;
			// Block Logic: Perform the dot product for element C[i][j], considering A is upper triangular.
			// The inner loop iterates from `k = 0` up to `j` because elements `a[k * N + i]` and `a[k * N + j]`
			// are non-zero only within this range due to A being upper triangular and then transposed.
			for (k = 0; k <= j; k++) {
				
				c[i * N + j] += a[k * N + i] * a[k * N + j];
				// Functional Utility: Exploit symmetry (C[j][i] = C[i][j]) to avoid redundant calculations.
				c[j * N + i] = c[i * N + j];
			}
		}
	}
}

/**
 * @brief Implements a non-optimized matrix solver that computes a result C based on input matrices A and B.
 * The solver performs a sequence of matrix operations: BBt = B * B_transpose, ABBt = A * BBt, AtA = A_transpose * A,
 * and finally C = ABBt + AtA. This implementation uses basic, unoptimized matrix multiplication and addition routines.
 *
 * @param N The dimension of the square matrices.
 * @param A Pointer to the input matrix A (N x N).
 * @param B Pointer to the input matrix B (N x N).
 * @return A pointer to the newly allocated result matrix C (N x N). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	// Functional Utility: Indicate the usage of the non-optimized solver to the standard output for tracing or debugging.
	printf("NEOPT SOLVER\n");

	// Functional Utility: Allocate memory for result matrix C and intermediate matrices BBt, ABBt, and AtA.
	// Invariant: All allocated memory blocks are initialized to zero.
	double *C = calloc(N * N, sizeof(double));
	double *BBt = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));

	// Block Logic: Compute BBt = B * B_transpose using `normal_x_normal_transpose`.
	// Precondition: `B` is a valid N x N matrix, `BBt` is allocated memory.
	// Invariant: `BBt` stores the result of B multiplied by its transpose.
	normal_x_normal_transpose(N, B, BBt);
	// Block Logic: Compute ABBt = A * BBt using `upper_x_normal`, assuming A is upper triangular.
	// Precondition: `A` and `BBt` are valid N x N matrices, `ABBt` is allocated memory.
	// Invariant: `ABBt` stores the result of A (upper triangular) multiplied by BBt.
	upper_x_normal(N, A, BBt, ABBt);
	// Block Logic: Compute AtA = A_transpose * A using `upper_transpose_x_upper`, assuming A is upper triangular.
	// Precondition: `A` is a valid N x N matrix, `AtA` is allocated memory.
	// Invariant: `AtA` stores the result of A_transpose multiplied by A (both upper triangular).
	upper_transpose_x_upper(N, A, AtA);
	// Block Logic: Compute C = ABBt + AtA using `add`.
	// Precondition: `ABBt` and `AtA` are valid N x N matrices, `C` is allocated memory.
	// Invariant: `C` stores the final sum of ABBt and AtA.
	add(N, ABBt, AtA, C);

	// Functional Utility: Free dynamically allocated memory for intermediate matrices.
	free(BBt);
	free(ABBt);
	free(AtA);

	return C;
}
