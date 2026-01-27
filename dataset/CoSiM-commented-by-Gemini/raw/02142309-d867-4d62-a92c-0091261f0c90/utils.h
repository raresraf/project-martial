
/**
 * @file utils.h
 * @brief Utility header for common functions and structures used in matrix computation solvers.
 *
 * This header file defines data structures, macros, and function prototypes
 * that are shared across different matrix solver implementations (e.g., BLAS,
 * non-optimized, optimized). It includes functionalities for random number
 * generation, test case management, data generation, and input file parsing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @brief Type definition for a function pointer to a matrix solver.
 *
 * A solver function takes the matrix dimension (N), two input matrices (A, B),
 * and returns a pointer to the resulting matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief Generates a random double-precision floating-point number within a specified range.
 * @param limit The absolute maximum value for the random number (range will be -limit to +limit).
 * @return A random double value.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief Structure to hold parameters for a single test case.
 * @var test::seed
 *   The seed for the random number generator, ensuring reproducible test data.
 * @var test::N
 *   The dimension of the square matrices for the test case.
 * @var test::output_save_file
 *   The file path where the output matrix for this test case should be saved.
 */
struct test {
	int seed;
	int N;
	
	char output_save_file[100];
};

/**
 * @brief Function prototype for the matrix solver.
 * @param N The dimension of the square matrices.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return A pointer to the resulting matrix.
 */
double* my_solver(int, double *, double *);

/**
 * @brief Function prototype for running a test case with a given solver.
 * @param test The test case parameters.
 * @param Solver A function pointer to the solver to be tested.
 * @param time A pointer to a float where the execution time will be stored.
 * @return An integer indicating success (0) or failure (-1).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief Function prototype for freeing allocated matrix data.
 * @param data A pointer to a pointer of the double array to be freed.
 */
void free_data(double **);

/**
 * @brief Function prototype for generating random matrix data for a test case.
 * @param test The test case parameters.
 * @param data A pointer to a pointer where the generated matrix data will be stored.
 * @param N The dimension of the matrix to generate.
 * @return An integer indicating success (0) or failure (-1).
 */
int generate_data(struct test, double **, int);

/**
 * @brief Function prototype for reading test case parameters from an input file.
 * @param file_path The path to the input file.
 * @param num_tests A pointer to an integer where the number of tests found will be stored.
 * @param tests A pointer to a pointer of a `struct test` array where parsed test cases will be stored.
 * @return An integer indicating success (0) or failure (-1).
 */
int read_input_file(char *, int *, struct test **);
