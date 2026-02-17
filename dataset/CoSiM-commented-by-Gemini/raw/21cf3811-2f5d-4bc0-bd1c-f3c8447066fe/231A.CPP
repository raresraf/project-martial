/**
 * @file 231A.CPP
 * @brief Solution for a counting problem based on a consensus threshold.
 * @details This program determines how many problems from a given set will be
 * solved based on the agreement of three individuals. The condition for solving a
 * problem is that at least two of the three individuals must agree on the solution.
 *
 * @section
 *
 * Algorithm:
 * The approach is a direct simulation. It iterates through each problem one by one.
 * For each problem, it sums the boolean confidence flags of the three individuals.
 * If the sum is two or greater, a counter is incremented.
 *
 * - Time Complexity: O(N), where N is the total number of problems, as each problem is processed once.
 * - Space Complexity: O(1), as storage is limited to a few variables for input and counting.
 */
#include <iostream>

using namespace std;

int main() {

  int n;
  cin >> n;
  int res = 0;

  /**
   * Block Logic: Iterates through each of the 'n' problems to evaluate them individually.
   * Invariant: Before each iteration, 'res' holds the count of problems that met the
   *            consensus criteria from the problems processed so far.
   */
  for (int i = 0; i < n; i++) {
    int p, v, t;
    cin >> p >> v >> t;

    /**
     * Block Logic: Checks if the consensus condition is met for the current problem.
     * Pre-condition: 'p', 'v', and 't' hold the boolean confidence values (0 or 1)
     *                for the current problem.
     */
    if ((p == 1 && v == 1) || (p == 1 && t == 1) || (v == 1 && t == 1)) {
      res++;
    }
  }

  cout << res << endl;
  return 0;
}
