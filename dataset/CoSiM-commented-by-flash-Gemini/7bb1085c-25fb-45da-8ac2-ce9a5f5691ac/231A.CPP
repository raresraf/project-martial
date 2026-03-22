/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines how many problems a team of three friends will implement
 *        in a programming contest. A problem is implemented if at least two out of the three
 *        friends are sure about its solution.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the total number of problems.
 *   2. Initializes a counter `total2` to 0, which will store the number of problems implemented.
 *   3. Iterates `n` times, once for each problem:
 *      a. Reads three integers (`aux1`, `aux2`, `aux3`), each being 1 if a friend is sure
 *         about the solution, and 0 otherwise.
 *      b. Calculates `total` as the sum of `aux1`, `aux2`, and `aux3`.
 *      c. If `total` is 2 or greater, it means at least two friends are sure,
 *         so `total2` is incremented.
 *   4. After processing all problems, prints the final count `total2`.
 *
 * Time Complexity: O(n), where `n` is the number of problems, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <iostream>
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
  int n, total;
  int aux1, aux2, aux3, total2 = 0;
  cin >> n;
  /**
   * Block Logic: Iterates through each problem to determine if it will be implemented.
   * Invariant: `total2` stores the count of problems implemented so far.
   * Pre-condition: `n` is a positive integer representing the number of problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> aux1 >> aux2 >> aux3;
    // Calculates the number of friends who are sure about the current problem.
    total = aux1 + aux2 + aux3;
    /**
     * Conditional Logic: Checks if at least two friends are sure about the solution.
     * Pre-condition: `total` holds the sum of confidence levels for the current problem.
     */
    if (total >= 2) {
      total2++; // Increments the counter for implemented problems.
    }
  }

  // Outputs the total number of problems that will be implemented.
  cout << total2;
  return 0;
}
