/**
 * @file utils.h
 * @brief Common utilities and declarations for the matrix solver project.
 * @details This header file defines shared data structures, macros, and function prototypes
 * used by the different solver implementations (BLAS, optimized, non-optimized) and the
 * test/comparison utilities.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @typedef Solver
 * @brief A function pointer type for a generic solver function.
 * @details This allows different solver implementations (e.g., `my_solver` from
 * `solver_blas.c`, `solver_opt.c`) to be used interchangeably by the testing framework.
 *
 * @param int The dimension of the matrices.
 * @param double* Pointer to the first input matrix (A).
 * @param double* Pointer to the second input matrix (B).
 * @return A pointer to the resulting solution matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief Macro to generate a random double-precision number within a given range.
 * @param limit The upper bound of the absolute value of the random number. The range will be [-limit, limit].
 * @return A random double within the specified range.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief A structure to hold the parameters for a single test case.
 */
struct test {
	int seed;  /**< The seed for the random number generator to ensure reproducibility. */
	int N;     /**< The dimension of the square matrices for this test. */
	
	char output_save_file[100]; /**< A file path to save the output matrix. */
};

/**
 * @brief Prototype for the solver function to be implemented by different source files.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Prototype for a function that runs a specific test case with a given solver.
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Prototype for a function to free the memory allocated for matrix data.
 */
void free_data(double **);

/**
 * @brief Prototype for a function that generates the input matrices for a given test case.
 */
int generate_data(struct test, double **, int);

/**
 * @brief Prototype for a function that reads test configurations from an input file.
 */
int read_input_file(char *, int *, struct test **);