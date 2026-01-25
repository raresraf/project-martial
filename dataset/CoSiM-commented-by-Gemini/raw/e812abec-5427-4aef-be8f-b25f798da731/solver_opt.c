
#include "utils.h"
#define MIN(x, y) (x > y ? y : x)

double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *AtA = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));
	double *result = calloc(N * N, sizeof(double));
	for (register int i = 0; i < N; ++i) {
		register double *ABLine = &AB[i * N];
		register double *ALine = &A[i * N];
		for (register int j = 0; j <= N; ++j) {
			register double ABsum = 0.0;
			for (register int k = i; k <= N; ++k) {
				ABsum += *(ALine + k) * B[k * N + j]; 
			}
			ABLine[j] = ABsum;
		}
	}
	for (register int i = 0; i < N; ++i) {
		register double *AtALine = &AtA[i * N];
		for (register int j = 0; j <= N; ++j) {
			register double AtAsum = 0.0;
			int min = MIN(i, j);
			for (register int k = 0; k <= min; ++k) {
				AtAsum += A[k * N + i] * A[k * N + j];
			}
			AtALine[j] = AtAsum;
		}
	}
	for (register int i = 0; i < N; ++i) {
		register double *resultLine = &result[i * N];
		register double *ABLine = &AB[i * N];
		for (register int j = 0; j < N; ++j) {
			register double sum = 0.0;
			register double *BtLine = &B[j * N];
			for (register int k = 0; k < N; ++k) {
				sum += *(ABLine + k) * *(BtLine + k);
			}
			resultLine[j] = sum + AtA[i * N + j];
		}
	}
	free(AtA);
	free(AB);
	return result;
}
