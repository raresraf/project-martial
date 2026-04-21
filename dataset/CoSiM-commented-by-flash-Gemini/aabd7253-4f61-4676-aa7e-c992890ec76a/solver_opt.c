
/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * 
 * Functional Intent: Evaluates the matrix expression $Res = (A \times B) \times B^T + (A^T \times A)$ 
 * using manual performance tuning techniques. It optimizes memory access patterns 
 * via pointer arithmetic, utilizes CPU registers for hot-path variables, and 
 * provides specialized kernels for constrained matrix products to minimize 
 * redundant floating-point operations.
 * 
 * Domain: HPC, Performance Optimization, C Programming.
 */

#include "utils.h"
#define UPPER 1
#define LOWER -1
#define NORMAL 0

/**
 * transpose - Computes the transpose of a square matrix.
 * 
 * Optimization: Uses pointer arithmetic and register variables to minimize 
 * address calculation overhead during the coordinate swap.
 */
void transpose(int N, double **C, double *M) {
	register int i, j;
	for(i = 0; i < N; i ++) {
		register double *ptrC = &(*C)[i * N ];
		register double *ptrM = &M[i];
		for(j = 0; j < N; j++) {
			// Logic: Direct pointer dereference for high-throughput copy.
			*ptrC = *ptrM;
			ptrM += N;
			ptrC++;
		}
	}
}

/**
 * multiply_normal - General matrix-matrix multiplication kernel.
 * 
 * Optimization: Leverages row-major data layout. Traverses source A row-wise 
 * and source B column-wise using direct pointer increments to maximize 
 * cache hit rates and reduce ALU pressure from index logic.
 */
void multiply_normal(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			
			// Block Logic: Innermost dot product accumulation.
			for(k = 0; k < N; k++) {
				suma += *ptrA * *ptrB;
				ptrA++;	
				ptrB += N;	
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

/**
 * multiply_upper - Optimized multiplication for upper-triangular source A.
 */
void multiply_upper(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				// Logic: Prunes operations where A[i][k] is guaranteed to be zero.
				if(i <= k)
					suma += *ptrA * *ptrB;
				ptrA++;	
				ptrB += N;	
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

/**
 * multiply_lower_upper - Specialized kernel for product of Lower x Upper.
 */
void multiply_lower_upper(int N, double **C, double *A, double *B, int typeA, int typeB) {
	register double *ptrC = *C;
	register int i, j, k;
	for(i = 0; i < N; i++) {
		register double *pa = &A[i * N];
		for(j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *ptrA = pa;
			register double *ptrB = &B[j];
			for(k = 0; k < N; k++) {
				// Logic: Restricts dot product to the non-zero intersection of L and U.
				if(i >= k && k <= j)
					suma += *ptrA * *ptrB;
				ptrA++;	
				ptrB += N;	
			}
			*ptrC = suma;
			ptrC++;
		}
	}
}

/**
 * add - High-speed element-wise matrix addition.
 */
void add(int N, double **C, double *A, double *B) {
	register double *ptrC = *C;
	register double *ptrA = &A[0];
	register double *ptrB = &B[0];
	register int i;
	// Block Logic: Linear sweep across flattened matrix memory.
	for(i = 0; i < N * N; i++) {
		*ptrC = *ptrA + *ptrB;
		ptrA++;
		ptrB++;
		ptrC++;
	}
}

/**
 * my_solver - Optimized orchestrator for the matrix expression.
 */
double* my_solver(int N, double *A, double* B) {
	
	double *AB = calloc(N* N,sizeof(double));
	multiply_upper(N, &AB, A, B, UPPER, NORMAL);

	double *Btrans = malloc((N * N) * sizeof(double));
	transpose(N, &Btrans, B);

	double *ABBt = calloc(N * N,sizeof(double));
	multiply_normal(N, &ABBt, AB, Btrans, NORMAL, NORMAL);

	
	double *Atrans = malloc((N * N) * sizeof(double));
	transpose(N, &Atrans, A);
	
	double *AtA = calloc(N* N,sizeof(double));
	multiply_lower_upper(N, &AtA, Atrans, A, LOWER, UPPER);

	double *result = malloc((N * N) * sizeof(double));
	add(N, &result, ABBt, AtA);

	free(AB);
	free(Btrans);
	free(ABBt);
	free(Atrans);
	free(AtA);

	return result;
}
