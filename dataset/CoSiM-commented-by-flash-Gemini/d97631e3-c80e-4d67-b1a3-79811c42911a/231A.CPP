/**
 * @file 231A.CPP
 * @brief Solves a problem where 'n' problems are presented, and for each problem,
 *        three participants indicate their readiness (1 for ready, 0 for not ready).
 *        The goal is to count how many problems will be implemented (i.e., at least two
 *        participants are ready).
 *
 * Algorithm:
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter 'ans' to 0.
 * 3. Loop 'n' times (using 'n--' as a loop counter):
 *    a. Read the readiness status (p, v, t) for the three participants for the current problem.
 *    b. If the sum of their readiness statuses (p + v + t) is 2 or more, increment 'ans'.
 * 4. Print the final count 'ans'.
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

  // Block Logic: Declares variables for the number of problems (n),
  //              readiness status of three participants (p, v, t),
  //              and a counter for solvable problems (ans).
  int n;
  cin >> n;
  int p, v, t;
  int ans = 0;

  /**
   * Block Logic: Iterates 'n' times, once for each problem, to read participant
   * readiness and determine if the problem is solvable. The loop condition `n--`
   * decrements 'n' after each iteration until it becomes 0.
   *
   * Invariant: At the start of each iteration, 'n' represents the remaining problems to process,
   * and 'ans' holds the count of solvable problems found so far.
   */
  while (n--) {
    // Block Logic: Reads the readiness status for three participants for the current problem.
    cin >> p >> v >> t;
    /**
     * Block Logic: Checks if at least two participants are ready for the current problem.
     * Each participant's readiness (p, v, t) is either 0 or 1. Their sum represents
     * the total number of ready participants.
     *
     * Pre-condition: p, v, t are binary (0 or 1).
     * Invariant: If the sum is 2 or more, the problem is considered solvable, and 'ans' is incremented.
     */
    if (p + v + t >= 2)
      ans++;
  }

  // Block Logic: Prints the total count of problems that are deemed solvable, followed by a newline.
  cout << ans << endl;
}