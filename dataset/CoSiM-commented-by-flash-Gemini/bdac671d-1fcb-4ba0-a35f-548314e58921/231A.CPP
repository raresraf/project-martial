/**
 * @file 231A.CPP
 * @brief Solves a problem where 'n' problems are presented, and for each problem,
 *        three participants indicate their readiness (1 for ready, 0 for not ready).
 *        The goal is to count how many problems will be implemented (i.e., at least two
 *        participants are ready).
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter 's' to 0.
 * 3. Loop 'n' times:
 *    a. Read the readiness status (k, m, in) for the three participants for the current problem.
 *    b. If the sum of their readiness statuses (k + m + in) is greater than 1 (meaning 2 or 3), increment 's'.
 * 4. Print the final count 's'.
 *
 * Time Complexity: O(N) where N is the number of problems, as the program iterates through each problem once.
 * Space Complexity: O(1) as only a few variables are used to store problem-specific data.
 */

#include <iostream>

using namespace std;

/**
 * @brief Entry point of the program.
 * Functional Utility: Drives the problem-solving logic by reading input,
 * iterating through problems, applying the readiness condition, and
 * finally reporting the total count of solvable problems.
 * @return 0 on successful execution.
 */
int main() {
  // Block Logic: Declares an integer 'n' for the total number of problems,
  //              'k', 'm', 'in' for participant readiness, and 's' as a counter for solvable problems.
  int n, k, m, in, s = 0;
  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, once for each problem, to read participant
   * readiness and determine if the problem is solvable.
   *
   * Invariant: At the start of each iteration, 'i' tracks the current problem index,
   * and 's' holds the count of solvable problems found so far. The loop continues
   * until 'i' reaches 'n'.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the readiness status for three participants (k, m, in) for the current problem.
    cin >> k >> m >> in;
    /**
     * Block Logic: Checks if at least two participants are ready for the current problem.
     * Each participant's readiness (k, m, in) is either 0 or 1. Their sum represents
     * the total number of ready participants.
     *
     * Pre-condition: k, m, in are binary (0 or 1).
     * Invariant: If the sum is greater than 1 (i.e., 2 or 3), the problem is considered solvable, and 's' is incremented.
     */
    if (k + m + in > 1) {
      s++;
    }
  }
  // Block Logic: Prints the total count of problems that are deemed solvable, followed by a newline.
  cout << s << endl;
  return 0;
}
