/**
 * @file 231A.CPP
 * @brief Solves a problem where 'n' problems are presented, and for each problem,
 *        three participants indicate their readiness (1 for ready, 0 for not ready).
 *        The goal is to count how many problems will be implemented (i.e., at least two
 *        participants are ready).
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter 'k' to 0.
 * 3. Loop 'n' times:
 *    a. Read the readiness status (a, b, c) for the three participants for the current problem.
 *    b. Calculate the sum of their readiness statuses 's'.
 *    c. If 's' is 2 or more, increment 'k'.
 * 4. Print the final count 'k'.
 *
 * Time Complexity: O(N) where N is the number of problems, as the program iterates through each problem once.
 * Space Complexity: O(1) as only a few variables are used to store problem-specific data.
 */

#include <stdio.h>
/**
 * @brief Entry point of the program.
 * Functional Utility: Drives the problem-solving logic by reading input,
 * iterating through problems, applying the readiness condition, and
 * finally reporting the total count of solvable problems.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares an integer 'n' for the total number of problems
  //              and 'k' as a counter for solvable problems, initialized to 0.
  int n, k = 0;
  // Block Logic: Reads the total number of problems 'n' from standard input.
  scanf("%d", &n);
  /**
   * Block Logic: Iterates 'n' times, once for each problem, to read participant
   * readiness and determine if the problem is solvable. The loop condition `n--`
   * decrements 'n' after each iteration until it becomes 0.
   *
   * Invariant: At the start of each iteration, 'n' represents the remaining problems to process,
   * and 'k' holds the count of solvable problems found so far.
   */
  while (n--) {
    // Block Logic: Declares integers 'a', 'b', and 'c' for the readiness status of three participants.
    int a, b, c;
    // Block Logic: Reads the readiness status for three participants for the current problem.
    scanf("%d %d %d", &a, &b, &c);
    // Block Logic: Calculates the sum of the readiness statuses of the three participants.
    int s = a + b + c;
    /**
     * Block Logic: Checks if at least two participants are ready for the current problem.
     * Each participant's readiness (a, b, c) is either 0 or 1. Their sum 's' represents
     * the total number of ready participants.
     *
     * Pre-condition: a, b, c are binary (0 or 1).
     * Invariant: If the sum 's' is 2 or more, the problem is considered solvable, and 'k' is incremented.
     */
    if (s >= 2)
      k++;
  }
  // Block Logic: Prints the total count of problems that are deemed solvable.
  printf("%d", k);
  return 0;
}
