
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;
	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    double *D = (double*) calloc(N * N, sizeof(double));
	if (D == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}

	
    for (i = 0; i < N; i++) {
		register double *p_c = C + i * N;
		register double *p_a = A + i * N;
		for (k = i; k < N; k++) {
			register double *p_b = B + k * N;
            for (j = 0; j < N; j++) {
				*(p_c + j) += *(p_a + k) * *(p_b + j);
            }
        }
    }
	
    for (i = 0; i < N; i++) {
		register double *p_c = C + i * N;
		register double *p_d = D + i * N;
        for (j = 0; j < N; j++) {
			register double rez = 0.0;
			register double *p_b = B + j * N;
            for (k = 0; k < N; k++) {
				rez += *(p_c + k) * *(p_b + k);
            }
			*(p_d + j) = rez;
        }
    }
	
	for (k = 0; k < N; k++) {
		register double *p_a = A + k * N;
		for (i = k; i < N; i++) {
			register double *p_d = D + i * N;
            for (j = k; j < N; j++) {
				*(p_d + j) += *(p_a + i) * *(p_a + j);
            }
		}
	}
	free(C);
	return D;
}
