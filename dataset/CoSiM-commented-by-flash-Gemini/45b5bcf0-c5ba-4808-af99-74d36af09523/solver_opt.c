
#include "utils.h"


double* my_solver(int N, double *A, double* B) {

	double *first_mul = calloc (N * N, sizeof(double));

	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));

	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));

	if (!third_mul)
		return NULL;

	double *result = malloc (N * N * sizeof(double));

	register int i, j, k;

	for (i = 0; i < N; i++) {
		double *pa_first = &A[i * (N + 1)];
		double *res = &first_mul[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &B[i * N + j];

			register double suma = 0;

			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}

			*res = suma;
			res++;
		}
	}

	for (i = 0; i < N; i++) {
		double *pa_first = &first_mul[i * N];
		double *res = &second_mul[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &B[j * N];

			register double suma = 0;

			for (k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb++;
			}

			*res = suma;
			res++;
		}
	}

	for (i = 0; i < N; i++) {
		double *pa_first = &A[i];
		double *res = &third_mul[i * N];

		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &A[j];

			register double suma = 0;

			for (k = 0; k <= i; k++) {
				suma += *pa * *pb;
				pa += N;
				pb += N;
			}

			*res = suma;
			res++;
		}
	}

	for (i = 0; i < N; i++) {
		register double *res = &result[i * N];
		register double *pa = &second_mul[i * N];
		register double *pb = &third_mul[i * N];

		for (j = 0; j < N; j++) {
			*res = *pa + *pb;
			res++;
			pa++;
			pb++;
		}
		res += N;
		pa += N;
		pb += N;
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);

	return result;
}
