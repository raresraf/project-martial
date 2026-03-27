/**
 * @file 231A.CPP
 * @brief Solves a problem where 'n' problems are presented, and for each problem,
 *        three participants indicate their readiness (1 for ready, 0 for not ready).
 *        The goal is to count how many problems will be implemented (i.e., at least two
 *        participants are ready).
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter 'sum' to 0.
 * 3. Loop 'n' times:
 *    a. Read the readiness status (p, v, t) for the three participants for the current problem.
 *    b. If the sum of their readiness statuses (p + v + t) is 2 or more, increment 'sum'.
 * 4. Print the final count 'sum'.
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
  //              'p', 'v', 't' for participant readiness, 'i' for the loop counter,
  //              and 'sum' as a counter for solvable problems, initialized to 0.
  int n, p, v, t, i, sum = 0;
  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, once for each problem, to read participant
   * readiness and determine if the problem is solvable.
   *
   * Invariant: At the start of each iteration, 'i' tracks the current problem index,
   * and 'sum' holds the count of solvable problems found so far. The loop continues
   * until 'i' reaches 'n'.
   */
  for (i = 0; i < n; i++) {
    // Block Logic: Reads the readiness status for three participants (p, v, t) for the current problem.
    cin >> p >> v >> t;
    /**
     * Block Logic: Checks if at least two participants are ready for the current problem.
     * Each participant's readiness (p, v, t) is either 0 or 1. Their sum represents
     * the total number of ready participants.
     *
     * Pre-condition: p, v, t are binary (0 or 1).
     * Invariant: If the sum is 2 or more, the problem is considered solvable, and 'sum' is incremented.
     */
    if (p + v + t >= 2) {
      sum = sum + 1;
    }
  }
  // Block Logic: Prints the total count of problems that are deemed solvable.
  cout << sum;
  return 0;
}
