
/**

 * @file solver_blas.c

 * @brief A matrix equation solver using BLAS routines.

 *

 * This file implements a solver for a matrix equation, likely of the form

 * C = A*B*B' + A'*A, where A and B are square matrices. The implementation

 * heavily relies on the CBLAS interface for high-performance matrix operations.

 */

#include "utils.h"



#include <cblas.h>



#include <string.h>





/**

 * @brief Solves a matrix equation using a sequence of BLAS operations.

 *

 * This function computes the solution to a complex matrix equation, which appears

 * to be C = A*B*B' + A'*A. It uses temporary matrices and leverages BLAS

 * functions for efficient computation of matrix copies, triangular matrix

 * multiplications, and general matrix-matrix multiplications.

 *

 * @param N The dimension of the square matrices A and B.

 * @param A A pointer to the first input matrix (N x N).

 * @param B A pointer to the second input matrix (N x N).

 * @return A pointer to the resulting matrix C, or NULL if memory allocation fails.

 */

double * my_solver(int N, double * A, double * B) {



    double * C;

    double * AB;



    // Functional Utility: Allocates memory for the result matrix C.

    C = calloc(N * N, sizeof( * C)); 

    if (C == NULL) {

        return NULL;

    }



    // Functional Utility: Allocates memory for the intermediate product AB.

    AB = calloc(N * N, sizeof( * C)); 

    if (AB == NULL) {

        return NULL;

    }



    

    // Logic: Copies matrix B into AB to preserve B for later use. AB will be modified.

    cblas_dcopy(

        N * N,

        B, 1,

        AB, 1

    );



    // Algorithm: Computes AB = A * AB, where A is an upper triangular matrix.

    // This is the first step in calculating the A*B term.

    cblas_dtrmm(

        CblasRowMajor,

        CblasLeft,

        CblasUpper,

        CblasNoTrans,

        CblasNonUnit,

        N, N,

        1.0, A, N,

        AB, N

    );



    

    // Logic: Copies matrix A into C. C will then be used to compute A'*A.

    cblas_dcopy(

        N * N,

        A, 1,

        C, 1

    );



    // Algorithm: Computes C = A' * C, which is equivalent to C = A'*A since C was a copy of A.

    cblas_dtrmm(

        CblasRowMajor,

        CblasLeft,

        CblasUpper,

        CblasTrans,

        CblasNonUnit,

        N, N,

        1.0, A, N,

        C, N

    );

    

    // Algorithm: Computes C = 1.0 * (AB) * B' + 1.0 * C.

    // This completes the computation C = A*B*B' + A'*A.

    cblas_dgemm(

        CblasRowMajor,

        CblasNoTrans,

        CblasTrans,

        N, N, N,

        1.0, AB, N,

        B, N, 1.0, C, N

    );



    // Functional Utility: Frees the memory used for the intermediate matrix AB.

    free(AB);



    return C;

}
