
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	register int i, j, k;
	register double *orig_p, *p1, *p2, sum;
	double *AB, *ABBt, *AtA, *C;

	
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		exit(-1);
	ABBt = calloc(N * N, sizeof(double));
	if (ABBt == NULL)
		exit(-1);	
	AtA = calloc(N * N, sizeof(double));
	if (AtA == NULL)
		exit(-1);	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(-1);

	
	for (i = 0; i < N; i++) {
		orig_p = A + i * N;
		for (j = 0; j < N; j++) {
			p1 = orig_p + i;
			p2 = &B[i * N + j];
			sum = 0.0;
			for (k = i; k < N; k++) {
				sum += *p1 * *p2;
				p1++;
				p2 += N;
			}
			AB[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		orig_p = AB + i * N;
		for (j = 0; j < N; j++) {
			p1 = orig_p;
			p2 = &B[j * N];
			sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += *p1 * *p2;
				p1++;
				p2++;
			}
			ABBt[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		orig_p = A + i;
		for (j = 0; j < N; j++) {
			p1 = orig_p;
			p2 = &A[j];
			sum = 0.0;
			for (k = 0; k < N; k++) {
				sum += *p1 * *p2;
				p1 += N;
				p2 += N;
				if (k == i || k == j)
					break;
			}
			AtA[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;	
}
