
/**
 * @file compare.c
 * @brief A utility to compare two matrices stored in binary files.
 *
 * This program memory-maps two files, treats them as square matrices of doubles,
 * and compares them element by element to check if they are equal within a
 * given precision. It is primarily used for verifying the results of matrix
 * computation algorithms.
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
 * @def check_err(a, b, err)
 * @brief Checks if two double-precision numbers are within a certain tolerance.
 *
 * @param a The first number.
 * @param b The second number.
 * @param err The tolerance.
 * @return 0 if the absolute difference is within the tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two matrices from files for equality within a given precision.
 *
 * This function opens two files, memory-maps them as matrices of doubles, and
 * then iterates through each element to check if the difference is within the
 * specified precision. It assumes the matrices are square and of the same size.
 *
 * @param file_path1 Path to the first matrix file.
 * @param file_path2 Path to the second matrix file.
 * @param precision The maximum allowed difference between corresponding elements.
 * @return 0 if the matrices are considered equal, -1 on error or if they differ.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Functional Utility: Opens file descriptors for the two matrix files for reading.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Functional Utility: Retrieves file status to check their sizes.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Pre-condition: The files must be of the same size to be comparable as matrices.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Functional Utility: Memory-maps the first file into the address space for direct memory access.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Functional Utility: Memory-maps the second file.
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Logic: Calculates the dimension N of the square matrix.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Iterates through each element of the matrices for comparison.
	// Invariant: `ret` remains 0 as long as all compared elements are within the tolerance.
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			// Inline: Compares two corresponding matrix elements.
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	// Functional Utility: Unmaps the files from memory and closes the file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main entry point for the matrix comparison utility.
 *
 * Parses command-line arguments for two file paths and a precision tolerance,
 * then calls cmp_files to perform the comparison and prints the result.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return 0 on success (matrices are equal), -1 on failure or if they differ.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	// Pre-condition: Validates that the correct number of command-line arguments is provided.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Logic: Parses the precision tolerance from the command-line arguments.
	sscanf(argv[3], "%lf", &precision);

	// Functional Utility: Invokes the file comparison function.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Output: Reports whether the matrices are identical or not.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
