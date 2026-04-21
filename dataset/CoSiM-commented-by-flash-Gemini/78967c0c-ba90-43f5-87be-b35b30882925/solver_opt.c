/**
 * @78967c0c-ba90-43f5-87be-b35b30882925/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 *
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using 
 * register-based accumulation and loop invariant code motion to minimize 
 * memory store operations and improve execution throughput.
 *
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Promotes induction variables and intermediate sum accumulators 
 * to hardware registers to reduce memory traffic and enhance computational density.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *At;
	double *Bt;
	double *C;
	
	double *Aux1;
	double *Aux2;
	
	// Synchronization: Induction variables in registers.
	register int i, j, k;
	
	// Pre-condition: Allocates heap workspace for intermediate calculations.
	At = calloc(N * N, sizeof(double));
	Bt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	Aux1 = calloc(N * N, sizeof(double));
	Aux2 = calloc(N * N, sizeof(double));
	
	/**
	 * Block Logic: Compute A^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			At[j * N + i] = A[i * N + j];
		}
	}
	
	/**
	 * Block Logic: Compute B^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Bt[j * N + i] = B[i * N + j];
		}
	}
	
	/**
	 * Block Logic: Compute Stage 1: Aux1 = A * B.
	 * Optimization: Local register `suma` minimizes store frequency to Aux1.
	 * Invariant: k starts at i to exploit A's upper triangularity.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			
			for (k = i; k < N; k++) {
				suma += A[i * N + k] * B[k * N + j];
			}
			
			Aux1[i * N + j] = suma;
		}
	}
	
	/**
	 * Block Logic: Compute Stage 2: Aux2 = Aux1 * B^T.
	 * Optimization: Scalar register accumulation for dot products.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			
			for (k = 0; k < N; k++) {
				suma += Aux1[i * N + k] * Bt[k * N + j];
			}
			
			Aux2[i * N + j] = suma;
		}
	}
	
	// Logic: Explicit buffer reset for reuse.
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Aux1[i * N + j] = 0;
		}
	}

	/**
	 * Block Logic: Compute Stage 3: Aux1 = A^T * A.
	 * Optimization: Leverages register `suma` for Gramian term calculation.
	 * Invariant: Exploits triangular sparsity by bounding k at i.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			
			for (k = 0; k <= i; k++) {
				suma += At[i * N + k] * A[k * N + j];
			}
			
			Aux1[i * N + j] = suma;
		}
	}
	
	/**
	 * Block Logic: Final summation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = Aux1[i * N + j] + Aux2[i * N + j];
		}
	}
	
	/**
	 * Cleanup: Deallocates temporary matrix buffers.
	 */
	free(At);
	free(Bt);
	
	free(Aux1);
	free(Aux2);
	
	return C;
}
