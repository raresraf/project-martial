
#include "utils.h"


double* my_solver(int N, double *A, double* B) {

	double* b_transpus = calloc(N * N, sizeof(double));
	double* a_transpus = calloc(N * N, sizeof(double));

	for (int i = 0; i < N; ++i ){
       for (int j = 0; j < N; ++j ){

          int index1 = i*N+j;

          int index2 = j*N+i;

          b_transpus[index2] = B[index1];
          a_transpus[index2] = A[index1];

       }
    }

    double* resultat = calloc(N * N, sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++){
            	if(k >= i){
                	sum = sum + A[i * N + k] * B[k * N + j];
                }
            }
            resultat[i * N + j] = sum;
        }
    }


    double* resultat2 = calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++){
                sum = sum + resultat[i * N + k] * b_transpus[k * N + j];
            }
            resultat2[i * N + j] = sum;
        }
    }


    double* resultat3 = calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++){
            	if(i >= k){
                	sum = sum + a_transpus[i * N + k] * A[k * N + j];
                }
            }
            resultat3[i * N + j] = sum;
        }
    }

    double *resultat4 = calloc(N * N, sizeof(double));

    for(int i = 0; i < N * N; i++){
    	resultat4[i] = resultat3[i] + resultat2[i];
    }

    free(resultat3);
    free(resultat2);
    free(resultat);
    free(a_transpus);
    free(b_transpus);

	return resultat4;
}
