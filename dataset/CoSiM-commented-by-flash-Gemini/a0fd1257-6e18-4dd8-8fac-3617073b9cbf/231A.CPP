/**
 * @file 231A.CPP
 * @brief This program determines the number of problems that can be implemented.
 * For each problem, three friends provide their opinion (1 if sure, 0 if not sure).
 * A problem is implemented if at least two friends are sure about the solution.
 *
 * Algorithm: Iterative problem evaluation based on a simple majority rule.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1)
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * Functional Utility: Optimizes standard input/output operations by unsynchronizing C++ streams
   * from C standard streams and unties cin from cout, improving performance for competitive programming.
   */
  ios::sync_with_stdio(0);
  cin.tie(0);

  // Functional Utility: 'n' stores the total number of problems to evaluate.
  int n;
  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;

  // Functional Utility: 'probs' accumulates the count of problems that can be implemented.
  int probs = 0;

  /**
   * Block Logic: Iterates through each problem to evaluate if it can be implemented.
   * Invariant: At the start of each iteration, 'i' represents the number of problems already evaluated.
   */
  for (int i = 0; i < n; i++) {
    // Functional Utility: 'p', 'v', 't' temporarily store the opinions of the three friends for the current problem.
    int p, v, t;
    // Block Logic: Reads the opinions of the three friends for the current problem.
    cin >> p >> v >> t;

    /**
     * Block Logic: Checks if at least two friends are sure about the solution for the current problem.
     * Pre-condition: 'p', 'v', 't' hold the opinions (0 or 1).
     * Invariant: If the sum of opinions is 2 or more, 'probs' (total implementable problems) is incremented.
     */
    if (p + v + t >= 2) {
      probs++;
    }
  }

  // Output the total number of problems that can be implemented.
  cout << probs << "
";

  return 0;
}
