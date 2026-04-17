/**
 * @file solver_opt.c
 * @brief Register-optimized matrix solver leveraging cache lines and pointer traversal.
 */
#include "utils.h"

int min(int a, int b) {
	return (a < b) ? a : b;
}


/**
 * @brief Highly optimized matrix combination solver.
 * @param N Size of square matrices.
 * @param A Input Matrix A.
 * @param B Input Matrix B.
 * @return Computed Matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	register int i, j, k;

	
	double *BB_T = calloc(N * N, sizeof(double));
	if(BB_T == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	double *C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	/**
	 * @pre Memory buffers BB_T and C are available.
	 * @post Populates BB_T symmetric product matrix using register offsets.
	 */
	for(i = 0; i < N; i++) {
		/// Cache efficient pointer traversal over base row.
		register double *orig_pa = &B[i * N];
		for(j = i; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &B[j * N];
			register double sum = 0.0;
			for(k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			BB_T[i * N + j] = sum;
			if(i != j)
				BB_T[j * N + i] = sum;
		}
	}

	/**
	 * @pre Symmetric B combinations processed.
	 * @post Matrix C filled with subsequent operations across A blocks.
	 */
	for(i = 0; i < N; i++) {
		/// Precalculated row origins avoiding repeated arithmetic.
		register double *orig_pa = &A[i * N + i];
		for(j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &BB_T[j + i * N];
			register double sum = 0.0;
			for(k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = sum; 
		}
	}

	/**
	 * @pre Base C products generated.
	 * @post Final inner combinations layered into C outputs.
	 */
	for(i = 0; i < N; i++) {
		register double *orig_pa = &A[i];
		for(j = i; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &A[j];
			register double sum = 0.0;
			for(k = 0; k <= min(i, j); k++) {
				sum += (*pa) * (*pb);
				pa += N;
				pb += N;

			}
			C[i * N + j] += sum;
			if(i != j)
				C[j * N + i] += sum;
		}
	}

	free(BB_T);

	return C;	
}
