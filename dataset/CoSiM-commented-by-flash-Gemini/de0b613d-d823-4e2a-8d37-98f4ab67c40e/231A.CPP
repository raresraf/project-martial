// 231A - Team

/**
 * @file 231A.CPP
 * @brief Solution to the "Team" competitive programming problem (Codeforces 231A).
 * This program determines how many problems a team can implement. For each problem,
 * three friends indicate if they are sure about the solution (1) or not (0).
 * A problem is implemented if at least two friends are sure about it.
 * Algorithm: Iterative counting of problems where the sum of sure votes is greater than or equal to 2.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
#include <iostream>

using namespace std;

int main() {
  // Declares 'n' for the total number of problems.
  // Declares 'p', 'v', 't' for individual friend's opinions (presumably binary: 0 or 1).
  // Declares 'sum' to count the number of problems that can be implemented, initialized to 0.
  int n, p, v, t, sum = 0;

  // Reads the total number of problems 'n'.
  cin >> n;

  /**
   * Block Logic: Iterates 'n' times (or until 'n' becomes 0), once for each problem.
   * Invariant: At the start of each iteration, 'n' represents the remaining problems to process,
   * and 'sum' holds the count of implementable problems found so far.
   * Pre-condition: 'n' is a positive integer indicating the number of problems to follow.
   * Post-condition: 'sum' holds the total count of implementable problems.
   */
  while (n--) {
    // Reads the opinions of the three friends for the current problem.
    // Each opinion (p, v, t) is expected to be 0 or 1.
    cin >> p >> v >> t;

    /**
     * Conditional Logic: Checks if at least two friends are sure about the solution.
     * If 'p', 'v', 't' are 0 or 1, their sum will be >= 2 if and only if
     * at least two of them are 1 (i.e., at least two friends are sure).
     */
    if (p + v + t >= 2)
      sum++; // Increments the counter for implementable problems.
  }

  // Prints the final count of implementable problems.
  cout << sum;

  return 0; // Standard practice to indicate successful execution.
}
