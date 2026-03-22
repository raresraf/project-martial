/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines how many problems a team of three friends will implement
 *        in a programming contest. A problem is implemented if at least two out of the three
 *        friends are sure about its solution. This version uses a concise ternary operator
 *        for conditional counting.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the total number of problems.
 *   2. Initializes a counter `count` to 0, which will store the number of problems implemented.
 *   3. Iterates `n` times, once for each problem:
 *      a. Reads three integers (`p`, `v`, `t`), each being 1 if a friend is sure
 *         about the solution, and 0 otherwise.
 *      b. If the sum `p + v + t` is greater than or equal to 2, `count` is incremented.
 *         This is done concisely using a ternary operator.
 *   4. After processing all problems, prints the final `count`.
 *
 * Time Complexity: O(n), where `n` is the number of problems, due to the single loop.
 * Space Complexity: O(1) for storing a few integer variables.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Team" problem.
   *
   * Reads the number of problems `n`, then processes each problem to count
   * how many will be implemented based on friends' confidence levels.
   *
   * @return int Returns 0 upon successful execution.
   */
  int n; // Total number of problems.
  int count = 0; // Counter for problems that will be implemented.
  int p, v, t;   // Confidence levels for Petya, Vasya, and Tonya.
  cin >> n;      // Reads the number of problems.
  /**
   * Block Logic: Iterates `n` times to process each problem.
   * Invariant: `count` stores the number of problems implemented up to the current iteration.
   * Pre-condition: `n` is a non-negative integer.
   */
  while (n--) {
    cin >> p >> v >> t; // Reads confidence levels for the current problem.
    /**
     * Conditional Logic: Checks if at least two friends are sure about the current problem.
     * Inline: Uses a ternary operator to add 1 to `count` if the sum of confidence
     *         levels is 2 or more, otherwise adds 0.
     */
    count += ((p + v + t) >= 2) ? 1 : 0;
  }
  // Prints the total number of problems that will be implemented.
  cout << count;
  return 0;
}