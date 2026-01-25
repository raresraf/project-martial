
#include "utils.h"


double* my_solver(int N, double *A, double* B)
{
	double *res = malloc(N * N * sizeof(*res));
	double *tmp1 = malloc(N * N * sizeof(*res));
	double *tmp2 = calloc(N * N, sizeof(*res));

	if (res == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	if (tmp1 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	if (tmp2 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = i; j < N; j++) {
			tmp1[i * N + j] = 0.0;

			for (int k = 0; k < N; k++) {
				tmp1[i * N + j] += B[i * N + k] * B[j * N + k];

				tmp2[i * N + j] += A[k * N + i] * A[k * N + j];
			}

			tmp2[j * N + i] = tmp2[i * N + j];
			tmp1[j * N + i] = tmp1[i * N + j];
		}
	}

		
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[i * N + j] = 0.0;
			for (int k = i; k < N; k++) {
				res[i * N + j] += A[i * N + k] * tmp1[k * N + j];

			}
			res[i * N + j] += tmp2[i * N + j];
		}
	}

	free(tmp1);
	free(tmp2);

	return res;
}
