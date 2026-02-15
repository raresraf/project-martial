
/**
 * @file solver_opt.c
 * @brief This file implements an optimized matrix solver for specific matrix operations.
 * It leverages techniques such as explicit pointer arithmetic and register keyword usage
 * to enhance performance compared to a non-optimized baseline. The optimizations are
 * tailored for matrix multiplication involving normal, transposed, and upper triangular matrices.
 *
 * Algorithm: Optimized naive matrix operations with pointer arithmetic and register variables.
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
 * @brief Computes the product of a normal matrix with the transpose of another normal matrix, with optimizations.
 * This function calculates C = A * A_transpose, storing the result in a symmetric manner (C[i][j] = C[j][i]).
 * It uses pointer arithmetic for efficient memory access and exploits symmetry to reduce redundant calculations.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the input matrix A (N x N).
 * @param c Pointer to the output matrix C (N x N), which will store A * A_transpose.
 */
void normal_x_normal_transpose(int N, double *a, double *c) {

	int i, j, k;
	
	// Functional Utility: Initialize pointer to the beginning of matrix 'a' for row-wise traversal.
	// This avoids repeated multiplication `i * N` in the inner loop.
	double *pa = &a[0];

	// Block Logic: Iterate over the upper triangle and diagonal of the resultant matrix C.
	// Due to symmetry (C[i][j] = C[j][i]), only these elements need to be explicitly calculated.
	// Precondition: `a` and `c` are initialized N x N matrices.
	// Invariant: At the end of the outer loops, the upper triangular and diagonal elements of C are computed.
	for (i = 0; i < N; i++) {
		
		// Functional Utility: Initialize pointer for column-wise traversal of the transposed matrix 'a'.
		// This pointer `pa_t` effectively moves along the rows of the original matrix 'a' for the transpose multiplication.
		double *pa_t = &a[0];

		for (j = 0; j <= i; j++) {
			// Performance Optimization: Use a register variable to store the sum for faster accumulation.
			register double sum = 0.0;

			// Block Logic: Perform the dot product for element C[i][j].
			// The inner loop iterates through the columns of A and rows of A_transpose.
			// Precondition: `pa` points to the current row `i` of `a`, `pa_t` points to current row `j` of `a`.
			// Invariant: `sum` accumulates the sum of products for the (i, j) element.
			for (k = 0; k < N; k++) {
				// Functional Utility: Access elements using pointer arithmetic for performance.
				// Inline: `*(pa + k)` accesses `a[i * N + k]`.
				// Inline: `*(pa_t + k)` accesses `a[j * N + k]`.
				sum += *(pa + k) * *(pa_t + k);
			}

			// Functional Utility: Assign the computed sum to the current element in C.
			c[i * N + j] = sum;
			// Functional Utility: Exploit symmetry (C[j][i] = C[i][j]) to avoid redundant calculations.
			c[j * N + i] = sum;
			
			// Functional Utility: Advance `pa_t` to the next row (equivalent to `j * N` for the next `j`).
			pa_t += N;
		}
		
		// Functional Utility: Advance `pa` to the next row (equivalent to `i * N` for the next `i`).
		pa += N;
	}
}

/**
 * @brief Performs matrix multiplication where the first matrix is upper triangular, with optimizations.
 * This function computes C = A * B, where A is implicitly upper triangular and both B and C are normal matrices.
 * It uses pointer arithmetic for efficient memory access and considers the upper triangular nature of A
 * to skip unnecessary multiplications.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the input matrix A (N x N), implicitly upper triangular.
 * @param b Pointer to the input matrix B (N x N).
 * @param c Pointer to the output matrix C (N x N), which will store A * B.
 */
void upper_x_normal(int N, double *a, double *b, double *c) {

	int i, j, k;
	// Functional Utility: Initialize pointers to the beginning of matrices `a` and `c`.
	// These pointers will be used for efficient row-wise traversal.
	double *pa = &a[0];
	double *pc = &c[0];

	// Block Logic: Iterate through each row of the resulting matrix C.
	// Precondition: `a`, `b`, and `c` are initialized N x N matrices.
	// Invariant: After each iteration of the outer loop, one row of C is partially computed.
	for (i = 0; i < N; i++) {
		
		// Functional Utility: Adjust `pa` to point to the `i`-th element of the `i`-th row of `a`.
		// This skips the zero elements in the lower triangular part of `a` as `a` is upper triangular.
		pa += i;
		
		// Functional Utility: Initialize pointer `pb` to the `i`-th row of matrix `b`.
		// This pointer will traverse the rows of `b` starting from row `i`.
		double *pb = &b[i * N];

		// Block Logic: Perform the dot product for the current row `i` of C.
		// This loop iterates through the columns `k` of matrix A (starting from `i`) and rows `k` of matrix B.
		// Precondition: `pa` points to the `i`-th element of current row `i` of `a`, `pb` points to current row `i` of `b`.
		// Invariant: The `i`-th row of `C` is accumulated based on multiplication with rows of `B` starting from `i`.
		for (k = i; k < N; k++) {
			// Performance Optimization: Use a register variable for `a[i*N+k]` for faster access.
			register double ra = *pa;

			// Block Logic: Accumulate the sum for each element in the current row of C.
			// Precondition: `ra` holds the value `a[i * N + k]`. `pc + j` points to `c[i * N + j]`. `pb` points to `b[k * N + j]`.
			// Invariant: `c[i * N + j]` is updated with `ra * b[k * N + j]`.
			for (j = 0; j < N; j++) {
				
				*(pc + j) += ra * *pb;
				
				// Functional Utility: Advance `pb` to the next element in the current row of `b`.
				pb++;
			}
			
			// Functional Utility: Advance `pa` to the next element in the current row of `a`.
			pa++;
		}
		
		// Functional Utility: Advance `pc` to the beginning of the next row of `c`.
		pc += N;
	}
}

/**
 * @brief Computes the product of an upper triangular matrix's transpose with another upper triangular matrix, with optimizations.
 * This function calculates C = A_transpose * A, where A is an implicitly upper triangular matrix.
 * The result is symmetric, and only the upper triangle and diagonal are explicitly computed and then mirrored.
 * It uses pointer arithmetic for efficient memory access and register variables for performance.
 *
 * @param N The dimension of the square matrices.
 * @param a Pointer to the input matrix A (N x N), implicitly upper triangular.
 * @param c Pointer to the output matrix C (N x N), which will store A_transpose * A.
 */
void upper_transpose_x_upper(int N, double *a, double *c) {

	int i, j, k;
	
	// Functional Utility: Store the original base pointer of matrix `a`.
	// This pointer `orig_pa` will be used to correctly calculate offsets within matrix `a`
	// for the columns `i` and `j` in the inner loop.
	double *orig_pa = &a[0];

	// Block Logic: Iterate over the upper triangle and diagonal of the resultant matrix C.
	// Due to symmetry (C[i][j] = C[j][i]), only these elements need to be explicitly calculated.
	// Precondition: `a` and `c` are initialized N x N matrices.
	// Invariant: At the end of the outer loops, the upper triangular and diagonal elements of C are computed.
	for (i = 0; i < N; i++) {

		for (j = 0; j <= i; j++) {
			
			// Performance Optimization: Use a register variable to store the sum for faster accumulation.
			register double sum = 0.0;
			// Functional Utility: Reset pointer `pa` to the beginning of matrix `a` for each (i, j) pair.
			// This allows traversal down the columns of `a` for the dot product.
			register double *pa = orig_pa;

			// Block Logic: Perform the dot product for element C[i][j], considering A is upper triangular.
			// The inner loop iterates from `k = 0` up to `j` because elements `a[k * N + i]` and `a[k * N + j]`
			// are non-zero only within this range due to A being upper triangular and then transposed effectively.
			for (k = 0; k <= j; k++) {
				// Functional Utility: Access elements using pointer arithmetic for performance.
				// Inline: `*(pa + i)` accesses `a[k * N + i]`
				// Inline: `*(pa + j)` accesses `a[k * N + j]`
				sum += *(pa + i) * *(pa + j);
				
				// Functional Utility: Advance `pa` to the beginning of the next row.
				pa += N;
			}

			// Functional Utility: Assign the computed sum to the current element in C.
			c[i * N + j] = sum;
			// Functional Utility: Exploit symmetry (C[j][i] = C[i][j]) to avoid redundant calculations.
			c[j * N + i] = sum;
		}
		
	}
}

/**
 * @brief Implements an optimized matrix solver that computes a result C based on input matrices A and B.
 * This solver performs a sequence of optimized matrix operations:
 * 1. BBt = B * B_transpose using `normal_x_normal_transpose`.
 * 2. ABBt = A * BBt using `upper_x_normal`.
 * 3. AtA = A_transpose * A using `upper_transpose_x_upper`.
 * 4. C = ABBt + AtA using `add`.
 * This optimized version aims to improve performance by leveraging specific matrix properties
 * (e.g., upper triangular, symmetry) and using low-level pointer arithmetic for memory access.
 *
 * @param N The dimension of the square matrices.
 * @param A Pointer to the input matrix A (N x N).
 * @param B Pointer to the input matrix B (N x N).
 * @return A pointer to the newly allocated result matrix C (N x N). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	// Functional Utility: Indicate the usage of the optimized solver to the standard output for tracing or debugging.
	printf("OPT SOLVER\n");

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
