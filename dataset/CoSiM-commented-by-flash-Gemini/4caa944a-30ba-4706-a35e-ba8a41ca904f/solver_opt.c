
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	double *C, *AtA, *AB, *ABBt;

	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBt = calloc(N * N, sizeof(double));
	DIE(ABBt == NULL, "calloc ABBt");

	
	for (i = 0; i < N; i++) {
		register double *orig_pa = A + i * N;
        for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = B + i * N + j;
			register double sum = 0.0;
            for (k = i; k < N; k++) {
                sum += *pa * *pb;
				pa++;
				pb += N;
            }
			*(AB + i * N + j) = sum;
        }
    }

	
	for (i = 0; i < N; i++) {
		double *orig_pab = AB + i * N;
        for (j = 0; j < N; j++) {
			register double *pab = orig_pab;
			register double *pb = B + j * N;
			register double sum = 0.0;
            for (k = 0; k < N; k++) {
				sum += *pab * *pb;
				pab++;
				pb++;
            }
			*(ABBt + i * N + j) = sum;
        }
    }

	
	
	for (i = 0; i < N; i++) {
		register double *orig_pat = A + i;
        for (j = 0; j < N; j++) {
			register double *pat = orig_pat;
			register double *pa = A + j;
			register double sum = 0.0;
            for (k = 0; k <= i && k <= j; k++) {
                sum += *pat * *pa;
				pat += N;
				pa += N;
            }
			*(AtA + i * N + j) = sum;
			*(C + i * N + j) = *(ABBt + i * N + j) + *(AtA + i * N + j);
        }
    }

	free(AB);
    free(AtA);
    free(ABBt);

	return C;
}
