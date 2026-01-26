
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *A_star_B;
	double *B_tr;
	double *OP1;
	double *A_tr;
	double *OP2;
	double *RES;
	int numMatrixElems = N * N;
	register int i, j, k;

	A_star_B = calloc(numMatrixElems, sizeof(*A_star_B));
	B_tr = calloc(numMatrixElems, sizeof(*B_tr));
	OP1 = calloc(numMatrixElems, sizeof(*OP1));
	A_tr = calloc(numMatrixElems, sizeof(*A_tr));
	OP2 = calloc(numMatrixElems, sizeof(*OP2));
	RES = calloc(numMatrixElems, sizeof(*RES));

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
			pa += i;
		    register double *pb = &B[j];
		    pb += N * i;

		    register double sum = 0;

			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			A_star_B[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
    	for (j = 0; j < N; j++) {
    		B_tr[j * N + i] = B[i * N + j];
    	}
	}

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A_star_B[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
		    register double *pb = &B_tr[j];
		    register double sum = 0;

			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			OP1[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
    	for (j = 0; j < N; j++) {
    		A_tr[j * N + i] = A[i * N + j];
    	}
	}

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = &A_tr[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
		    register double *pb = &A[j];
		    register double sum = 0;

			for (k = 0; k <= i; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}

			OP2[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
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

