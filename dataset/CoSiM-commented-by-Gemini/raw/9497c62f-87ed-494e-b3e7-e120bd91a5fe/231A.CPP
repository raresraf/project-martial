/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve.
 *
 * This program solves the "Team" problem from Codeforces. It determines how
 * many problems a three-person team will decide to implement based on their
 * confidence. The team solves a problem only if at least two of the three
 * members are confident about the solution.
 *
 * @algorithm Simulation / Counting
 * @complexity
 * - Time: O(N), where N is the number of problems, as it iterates through each
 *   problem once.
 * - Space: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // t: the number of problems.
  // a, b, c: confidence of the three team members (1 for sure, 0 otherwise).
  // d: temporary counter for members sure about the current problem.
  // e: total count of problems to be implemented.
  int t, a, b, c, d, e = 0;
  cin >> t;

  /**
   * @brief Block Logic: Iterates through each problem to check team confidence.
   * Invariant: At the end of each iteration, `e` holds the count of problems
   *            that met the criteria from the subset of problems processed so far.
   */
  while (t--) {
    cin >> a;
    cin >> b;
    cin >> c;
    d = 0; // Reset counter for each new problem.

    /**
     * @brief Block Logic: Counts how many members are sure about the current problem.
     * Note: Since the inputs are only 0 or 1, this can be simplified to `d = a + b + c;`
     */
    if (a == 1) {
      d = d + 1;
    }
    if (b == 1) {
      d = d + 1;
    }
    if (c == 1) {
      d = d + 1;
    }

    // Pre-condition: Check if at least two members are confident.
    if (d >= 2) {
      e = e + 1;
    }
  }

  // Output the total number of problems the team will implement.
  cout << e;
}
