
#include "utils.h"


double* my_solver(int size, double *A, double* B) {
    
    register int i, j, k;
    register int N = size;
    double *AB = (double *) malloc(N * N * sizeof(double));
    for (i = 0; i < N; i++) {
        register double *line_a = A + i * N;
        for (j = 0; j < N; j++){
            register double *line_a_aux = line_a + i;
            register double *line_b = B + i * N + j;
            register double sum = 0;
            for (k = i; k < N; k++){
                sum += *line_a_aux * *line_b;
                line_a_aux++;
                line_b += N;
            }
            AB[i * N + j] = sum;
        }
    }

    double *ABBt = (double *) malloc(N * N * sizeof(double));
    for (i = 0; i < N; i++) {
        register double *line_ab = AB + i * N;
        for (j = 0; j < N; j++){
            register double *line_ab_aux = line_ab;
            register double *line_b_trans = B + j * N;
            register double sum = 0;
            for (k = 0; k < N; k++){
                sum += *line_ab_aux * *line_b_trans;
                line_ab_aux++;
                line_b_trans++;
            }
            ABBt[i * N + j] = sum;
        }
    }

    for (i = 0; i < N; i++) {
        for (k = 0; k <= i; k++) {
            for (j = 0; j < N; j++){
                ABBt[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }

    free(AB);
    return ABBt;
}
