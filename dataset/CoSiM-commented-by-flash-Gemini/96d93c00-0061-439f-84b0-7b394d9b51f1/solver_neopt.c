
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *A_star_B;
	double *B_tr;
	double *OP1;
	double *A_tr;
	double *OP2;
	double *RES;
	int numMatrixElems = N * N;

	A_star_B = calloc(numMatrixElems, sizeof(*A_star_B));
	B_tr = calloc(numMatrixElems, sizeof(*B_tr));
	OP1 = calloc(numMatrixElems, sizeof(*OP1));
	A_tr = calloc(numMatrixElems, sizeof(*A_tr));
	OP2 = calloc(numMatrixElems, sizeof(*OP2));
	RES = calloc(numMatrixElems, sizeof(*RES));

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if (k >= i) {
					A_star_B[i * N + j] += A[i * N + k] * B[k * N + j];
				}
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) {
    		B_tr[j * N + i] = B[i * N + j];
    	}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				OP1[i * N + j] += A_star_B[i * N + k] * B_tr[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) {
    		A_tr[j * N + i] = A[i * N + j];
    	}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if (j >= k) {
					OP2[i * N + j] += A_tr[i * N + k] * A[k * N + j];
				}
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			RES[i * N + j] = OP1[i * N + j] + OP2[i * N + j];
		}
	}

	free(A_star_B);
	free(B_tr);
	free(OP1);
	free(A_tr);
	free(OP2);

	return RES;
}
