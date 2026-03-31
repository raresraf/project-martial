/**
 * @file 231A.CPP
 * @brief C solution for the "Team" problem, counting solvable problems.
 *
 * This program, written in C, determines how many problems a three-person team
 * will solve. It first reads the total number of problems, `n`. Then, it reads
 * the confidence levels (0 or 1) of three members for each of the `n` problems,
 * storing them in a 2D array. A problem is considered solvable if at least two
 * members are confident. The program iterates through the stored data, counts
 * the solvable problems, and prints the total.
 *
 * @author f2e5f077-b884-4e4f-bcf7-60a4cad46c6e
 *
 * @algorithm Two-Pass Counting with Explicit Condition Checking
 * @complexity
 * Time: O(N), for iterating through the `n` problems twice (once for input,
 *       once for logic).
 * Space: O(N), to store the `n x 3` matrix of opinions.
 */
#include <stdio.h>

int main() {
  // `n`: number of problems. `i`: loop counter. `cou`: counter for solvable problems.
  int n, i, cou = 0;
  scanf("%d", &n);

  // `c[n][3]`: a Variable Length Array (VLA) to store all inputs.
  int c[n][3];

  /**
   * @brief First pass: Read all opinions into the 2D array.
   */
  for (i = 0; i < n; i++)
    scanf("%d %d %d", &c[i][0], &c[i][1], &c[i][2]);

  /**
   * @brief Second pass: Iterate through problems to count solvable ones.
   */
  for (i = 0; i < n; i++) {
    /**
     * @brief Condition to check if at least two members are confident.
     * This logic explicitly checks all three possible pairs of confident members.
     * It is equivalent to checking if `c[i][0] + c[i][1] + c[i][2] >= 2`.
     */
    if (c[i][0] == 1 && c[i][1] == 1 || c[i][0] == 1 && c[i][2] == 1 ||
        c[i][1] == 1 && c[i][2] == 1)
      cou++;
  }
  // Print the total count of solvable problems.
  printf("%d", cou);
  return 0;
}