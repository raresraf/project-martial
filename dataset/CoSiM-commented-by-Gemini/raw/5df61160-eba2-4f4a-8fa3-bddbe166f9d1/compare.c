/**
 * @file compare.c
 * @brief Memory-mapped binary file comparator for solver verification.
 * Evaluates binary outputs for matrix floating-point values against a specific tolerance precision.
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
 * @brief Validates if two double precision values are equal within an error margin.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Maps two matrix files into memory and verifies floating-point equivalence.
 * @param file_path1 Path to the baseline output file.
 * @param file_path2 Path to the test output file.
 * @param precision The acceptable epsilon delta for tolerance testing.
 * @return 0 on success (match), -1 on error or discrepancy.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	/**
	 * @pre Both files are successfully opened and stat structs populated.
	 * @post Verification terminates early if file sizes mismatch avoiding partial comparisons.
	 */
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	/**
	 * @pre File 1 size matches File 2, memory mapped.
	 * @post Returns error if mapping allocation fails for file 1.
	 */
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	/**
	 * @pre First file mapped successfully.
	 * @post Handles cleanup and returns error if mapping second file fails.
	 */
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * @pre Matrices are mapped to RAM. The matrix geometry is assumed N x N.
	 * @post All points compared up to acceptable threshold. Stops at first discrepancy.
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Entrypoint to parse runtime precision args and trigger the binary comparator.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	/**
	 * @pre Input arg vector contains at least file names and precision.
	 * @post Execution bounds are verified.
	 */
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	sscanf(argv[3], "%lf", &precision);

	ret = cmp_files(argv[1],argv[2],precision);
	
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
