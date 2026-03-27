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
 * 3. Loop 'n' times:
 *    a. Read the readiness status (p, v, t) for the three participants for the current problem.
 *    b. If the sum of their readiness statuses (p + v + t) is greater than 1 (meaning 2 or 3), increment 'ans'.
 * 4. Print the final count 'ans'.
 *
 * Time Complexity: O(N) where N is the number of problems, as the program iterates through each problem once.
 * Space Complexity: O(1) as only a few variables are used to store problem-specific data.
 */

#include <bits/stdc++.h>

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
  //              'ans' as a counter for solvable problems (initialized to 0),
  //              and 'p', 'v', 't' for participant readiness.
  int n, ans = 0, p, v, t;
  // Block Logic: Reads the total number of problems 'n' from standard input.
  scanf("%d", &n);
  /**
   * Block Logic: Iterates 'n' times, once for each problem, to read participant
   * readiness and determine if the problem is solvable.
   *
   * Invariant: At the start of each iteration, 'i' tracks the current problem index,
   * and 'ans' holds the count of solvable problems found so far. The loop continues
   * until 'i' reaches 'n'.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the readiness status for three participants (p, v, t) for the current problem.
    scanf("%d %d %d", &p, &v, &t);
    /**
     * Block Logic: Checks if at least two participants are ready for the current problem.
     * Each participant's readiness (p, v, t) is either 0 or 1. Their sum represents
     * the total number of ready participants.
     *
     * Pre-condition: p, v, t are binary (0 or 1).
     * Invariant: If the sum is greater than 1 (i.e., 2 or 3), the problem is considered solvable, and 'ans' is incremented.
     */
    if (p + v + t > 1)
      ans++;
  }
  // Block Logic: Prints the total count of problems that are deemed solvable.
  printf("%d", ans);
  return 0;
}