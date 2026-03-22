/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines how many problems a team of three friends will implement
 *        in a programming contest. A problem is implemented if at least two out of the three
 *        friends are sure about its solution.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the total number of problems.
 *   2. Initializes a counter `c` to 0, which will store the number of problems implemented.
 *   3. Iterates `n` times, once for each problem:
 *      a. Reads three integers (`p`, `v`, `t`), each being 1 if a friend is sure
 *         about the solution, and 0 otherwise.
 *      b. If the sum `p + v + t` is greater than 1 (meaning 2 or 3), it means at least two
 *         friends are sure, so `c` is incremented.
 *   4. After processing all problems, prints the final count `c`.
 *
 * Time Complexity: O(n), where `n` is the number of problems, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  /**
   * @brief Main function that solves the "Team" problem.
   *
   * Reads problem confidence levels from three friends and counts how many problems
   * will be implemented based on the rule that at least two friends must be sure.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n, k, c = 0; // `n` is number of problems, `c` is counter for implemented problems.
                   // `k` is declared but not used in this specific implementation.
  cin >> n;
  /**
   * Block Logic: Iterates through each of the `n` problems.
   * Invariant: `c` stores the number of problems implemented up to the current iteration.
   * Pre-condition: `n` is a non-negative integer representing the total number of problems.
   */
  for (int i = 0; i < n; i++) {
    int p, v, t; // Confidence levels for Petya, Vasya, and Tonya for the current problem.
    cin >> p >> v >> t;
    /**
     * Conditional Logic: Checks if the sum of confidence levels is greater than 1.
     * This implies at least two friends (out of three possible inputs of 0 or 1)
     * are sure about the problem.
     */
    if ((p + v + t) > 1)
      c++; // Increments the counter if the problem will be implemented.
  }
  // Outputs the total number of problems that will be implemented.
  cout << c;
  return 0;
}
