/**
 * @55bb74e1-cf06-4caa-bfc5-96a1b07e2538/231A.CPP
 * @brief This program determines the number of problems that can be implemented
 * by a team of three, given their individual opinions on solving the problem.
 * A problem is implemented if at least two team members agree to solve it.
 *
 * Algorithm: Iterative counting with conditional increment.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 */

#include <iostream>
using namespace std;

int main() {
  int n, x, count = 0;
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to evaluate team consensus.
   * Invariant: At the start of each iteration, 'count' holds the number of
   * successfully implemented problems from previous iterations.
   */
  for (int i = 0; i < n; i++) {
    int surety = 0;
    /**
     * Block Logic: Reads the opinions of the three team members for the current problem.
     * Invariant: 'surety' accumulates the number of affirmative votes (1s) for the current problem.
     */
    for (int i = 0; i < 3; i++) {
      cin >> x;
      // Inline: Increments 'surety' if the current team member agrees to solve the problem.
      if (x == 1)
        surety++;
    }

    /**
     * Block Logic: Determines if the current problem meets the implementation criteria.
     * Pre-condition: 'surety' contains the total number of affirmative votes.
     * Invariant: 'count' is incremented only if 'surety' is 2 or more.
     */
    if (surety >= 2)
      count++;
  }

  // Output the total number of problems that the team decided to implement.
  cout << count << endl;
  return 0;
}
