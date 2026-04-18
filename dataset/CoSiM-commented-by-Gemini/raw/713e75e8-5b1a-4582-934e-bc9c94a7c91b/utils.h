/**
 * @file utils.h
 * @brief A header file containing utility functions, type definitions, and data structures for a matrix solver testing framework.
 *
 * This file declares a standard interface for different solver implementations,
 * defines a structure for test cases, and provides forward declarations for
 * functions related to data generation, test execution, and file I/O.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 * @typedef Solver
 * @brief A function pointer type that represents a matrix solver.
 *
 * Any function that matches this signature can be used as a solver within the testing framework.
 * It takes the matrix dimension N and two input matrices (as double pointers) and returns
 * a pointer to the resulting matrix.
 */
typedef double* (*Solver)(int, double *, double*);

/**
 * @def get_rand_double(limit)
 * @brief A macro to generate a random double-precision number within a given range.
 *
 * @param limit The upper bound of the range. The generated number will be in [-limit, limit].
 * @return A random double value.
 */
#define get_rand_double(limit) ((((double)rand()) / RAND_MAX) * (2 * limit) - limit)

/**
 * @struct test
 * @brief A structure to hold the parameters for a single test case.
 */
struct test {
	int seed; ///< The seed for the random number generator, ensuring reproducibility.
	int N;    ///< The dimension of the square matrices for the test.
	
	char output_save_file[100]; ///< The path to the file where the output matrix should be saved.
};

/**
 * @brief A forward declaration for a solver implementation.
 *
 * This function is expected to be defined in one of the solver source files (e.g., solver_opt.c).
 * @param N The dimension of the matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the result matrix.
 */
double* my_solver(int, double *, double *);

/**
 * @brief A forward declaration for a function to run a test case.
 *
 * @param t The test structure containing the test parameters.
 * @param s A function pointer to the solver to be tested.
 * @param time A pointer to a float where the execution time will be stored.
 * @return An integer indicating the test result (e.g., 0 for success).
 */
int run_test(struct test, Solver, float *);

/**
 * @brief A forward declaration for a function to free allocated data.
 *
 * @param data A pointer to an array of matrix pointers to be freed.
 */
void free_data(double **);

/**
 * @brief A forward declaration for a function to generate test data.
 *
 * @param t The test structure containing parameters for data generation.
 * @param data A pointer to an array where the generated matrices will be stored.
 * @param num_matrices The number of matrices to generate.
 * @return An integer indicating the result of the operation.
 */
int generate_data(struct test, double **, int);

/**
 * @brief A forward declaration for a function to read test configuration from a file.
 *
 * @param file_path The path to the input file.
 * @param num_tests A pointer to an integer where the number of tests read will be stored.
 * @param tests A pointer to an array of test structures where the parsed tests will be stored.
 * @return An integer indicating the result of the operation.
 */
int read_input_file(char *, int *, struct test **);
