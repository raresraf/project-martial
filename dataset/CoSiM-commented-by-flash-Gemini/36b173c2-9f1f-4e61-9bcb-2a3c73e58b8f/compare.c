/**
 * @file compare.c
 * @brief Compares two files containing double-precision floating-point matrices for equality within a given precision.
 *
 * This program memory-maps two input files, interprets their content as square matrices
 * of double-precision floating-point numbers, and compares them element-wise.
 * It reports if the files differ in size or if any corresponding elements differ
 * by more than a specified precision (tolerance).
 *
 * Algorithm: Memory-mapped file comparison, element-wise double comparison with tolerance.
 * Time Complexity: O(N*N), where N is the dimension of the square matrix.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

/**
 * @brief Macro to check if two double values are equal within a specified error margin.
 *
 * This macro returns 0 if the absolute difference between 'a' and 'b' is less
 * than or equal to 'err', indicating they are approximately equal. Otherwise,
 * it returns -1.
 *
 * @param a (double): The first double value.
 * @param b (double): The second double value.
 * @param err (double): The maximum allowed absolute difference (precision).
 * @return (int): 0 if values are approximately equal, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two files as double-precision floating-point matrices.
 *
 * This function opens, memory-maps, and compares the contents of two files.
 * It assumes the files contain square matrices of doubles. Comparison is
 * performed element-wise using a given precision.
 *
 * @param file_path1 (const char*): Path to the first file.
 * @param file_path2 (const char*): Path to the second file.
 * @param precision (double): The tolerance for comparing floating-point numbers.
 * @return (int): 0 if files are identical within precision, -1 otherwise.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Block Logic: Opens the first file in read-only mode.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	// Block Logic: Opens the second file in read-only mode.
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Block Logic: Retrieves file status information for both files.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Block Logic: Checks if the sizes of the two files are identical.
	// Precondition: For a valid matrix comparison, file sizes must match.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Block Logic: Memory-maps the first file into the process's address space for direct access.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	// Block Logic: Error checking for memory-mapping the first file.
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Block Logic: Memory-maps the second file into the process's address space.
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	// Block Logic: Error checking for memory-mapping the second file.
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size); // Inline: Unmaps the first file's memory in case of error.
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Block Logic: Calculates the dimension N of the square matrix.
	// N = sqrt(total_bytes / size_of_double).
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Iterates through each element of the matrices for comparison.
	// Invariant: Iterates row by row, then column by column.
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			// Block Logic: Compares corresponding elements using the defined precision.
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision);
			// Block Logic: If a difference exceeding precision is found, report and jump to cleanup.
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done; // Inline: Jumps to the 'done' label for resource cleanup.
			}
		}
	}

done:
	// Block Logic: Unmaps the memory regions for both files.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	// Block Logic: Closes the file descriptors.
	close(fd1);
	close(fd2);

	return ret; // Functional Utility: Returns the comparison result (0 for OK, -1 for differences).
}

/**
 * @brief Main entry point of the comparison program.
 *
 * Parses command-line arguments for file paths and precision,
 * then calls `cmp_files` to perform the comparison and reports the result.
 *
 * @param argc (int): The number of command-line arguments.
 * @param argv (const char**): Array of command-line argument strings.
 * @return (int): 0 for successful comparison (files identical), -1 for differences or errors.
 */
int main(int argc, const char **argv)
{
	double precision;
	int ret = 0;

	// Block Logic: Checks for the correct number of command-line arguments.
	// Precondition: Requires three arguments: mat1_path, mat2_path, tolerance.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1); // Inline: Exits with an error code if arguments are insufficient.
	}

	// Block Logic: Parses the precision (tolerance) from the third command-line argument.
	sscanf(argv[3], "%lf", &precision);

	// Block Logic: Calls the file comparison function with provided arguments.
	ret = cmp_files(argv[1],argv[2],precision);

	// Block Logic: Prints the final comparison result to standard output.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret; // Functional Utility: Returns the comparison result.
}
