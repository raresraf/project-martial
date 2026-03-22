/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Team.
 *        This program determines how many problems a team of three friends will implement
 *        in a programming contest. A problem is implemented if at least two out of the three
 *        friends are sure about its solution.
 *
 * Algorithm:
 *   1. Reads an integer `n`, representing the total number of problems.
 *   2. Initializes a counter `cnt` to 0, which will store the number of problems implemented.
 *   3. Iterates `n` times, once for each problem:
 *      a. Reads three integers (`a`, `b`, `c`), each being 1 if a friend is sure
 *         about the solution, and 0 otherwise.
 *      b. If the sum `a + b + c` is 2 or greater, it means at least two friends are sure,
 *         so `cnt` is incremented.
 *   4. After processing all problems, prints the final count `cnt`.
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
  int n;
  cin >> n;

  int cnt = 0; // Initializes the counter for problems that will be implemented.
  int a, b, c;
  /**
   * Block Logic: Iterates through each of the `n` problems.
   * Invariant: `cnt` stores the number of problems implemented up to the current iteration.
   * Pre-condition: `n` is a non-negative integer representing the total number of problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c; // Reads confidence levels for the current problem.
    /**
     * Conditional Logic: Checks if at least two friends are sure about the current problem.
     * Pre-condition: `a`, `b`, `c` represent confidence levels (0 or 1).
     */
    if (a + b + c >= 2) {
      cnt++; // Increments the counter if the problem will be implemented.
    }
  }

  // Outputs the total number of problems that will be implemented.
  cout << cnt << endl;
  return 0;
}
