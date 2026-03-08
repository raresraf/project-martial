/**
 * @file 231A.CPP
 * @brief Counts the number of problems that will be implemented, where each problem requires at least two out of three judges to agree.
 *
 * Problem Name: Young Physicist (Codeforces 231A) - common competitive programming interpretation
 * Algorithm: Iterative processing of problems. For each problem, it sums three binary votes (0 or 1)
 *            and increments a counter if the sum is 2 or more.
 * Time Complexity: O(N) where N is the number of problems, due to two linear passes.
 * Space Complexity: O(N) for storing the votes of all problems.
 */
#include <stdio.h> // Required for C-style input/output operations (scanf, printf).

/**
 * @brief Main function to calculate the number of problems to be implemented.
 * @pre Input consists of an integer 'n' (total problems), followed by 'n' lines,
 *      each containing three binary integers (P, V, T) representing judge agreements (votes).
 * @post Prints the total count of problems that will be implemented.
 */
int main(void) {
  int n;     // Variable to store the total number of problems.
  int cnt = 0; // Counter for problems where at least two judges agree.

  // Arrays to store votes from three judges for up to 1000 problems.
  int P[1000];
  int V[1000];
  int T[1000];

  // Functional Utility: Reads the total number of problems.
  scanf("%d", &n);

  /**
   * Block Logic: Reads 'n' sets of three binary votes for each problem.
   * Invariant: After each iteration, P[i], V[i], T[i] hold the votes for the i-th problem.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &P[i], &V[i], &T[i]); // Reads three votes for the current problem.
  }

  /**
   * Block Logic: Iterates through each problem to check if it qualifies for implementation.
   * Invariant: 'cnt' increments each time the sum of votes for a problem (P[i] + V[i] + T[i])
   *            is greater than or equal to 2.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Checks the agreement condition for the current problem.
     * Pre-condition: P[i], V[i], T[i] hold the votes for the i-th problem.
     * Invariant: If the sum of votes is 2 or more, at least two judges agreed.
     */
    if (P[i] + V[i] + T[i] >= 2) { // Sum of votes >= 2 means at least two judges agreed.
      cnt++;                       // Increment the count of implemented problems.
    }
  }

  // Functional Utility: Prints the final count of problems to be implemented.
  printf("%d", cnt);

  // Indicates successful program execution.
  return 0;
}
