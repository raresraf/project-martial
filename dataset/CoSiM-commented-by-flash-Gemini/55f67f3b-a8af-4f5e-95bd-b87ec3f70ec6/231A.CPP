/**
 * @55f67f3b-a8af-4f5e-95bd-b87ec3f70ec6/231A.CPP
 * @brief This program calculates the number of problems a team can solve.
 * Each problem requires at least two out of three team members (represented by p, v, t)
 * to agree on solving it.
 *
 * Algorithm: Iterative counting of successful problem resolutions based on majority agreement.
 * Time Complexity: O(N) where N is the total number of problems.
 * Space Complexity: O(1) for storing input and counters.
 */

#include <iostream>

using namespace std;

int main() {
  int n; // Declares 'n', representing the total number of problems to evaluate.
  int p, v, t; // Declares 'p', 'v', 't' as binary indicators of agreement from three team members.
  int solve; // Declares 'solve', used to count the number of agreements for a single problem.
  int problems; // Declares 'problems', the global counter for successfully agreed-upon problems.

  cin >> n; // Reads the total number of problems from standard input.

  problems = 0; // Initializes the count of solvable problems to zero.
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   * Invariant: 'problems' accurately reflects the count of solvable problems
   * from previous iterations.
   */
  for (int i = 0; i < n; i++) {
    solve = 0; // Resets the agreement counter for the current problem.
    cin >> p >> v >> t; // Reads the agreement status (binary: 0 or 1) for the three team members.
    // Inline: Increments 'solve' if the first team member (p) agrees.
    if (p) {
      solve++;
    }
    // Inline: Increments 'solve' if the second team member (v) agrees.
    if (v) {
      solve++;
    }
    // Inline: Increments 'solve' if the third team member (t) agrees.
    if (t) {
      solve++;
    }
    /**
     * Block Logic: Determines if the current problem has enough agreement to be solved.
     * Pre-condition: 'solve' contains the total number of agreements for the current problem.
     * Invariant: 'problems' is incremented if 'solve' is 2 or more, indicating a majority.
     */
    if (solve >= 2) {
      problems++;
    }
  }
  // Prints the total number of problems that the team collectively decided to solve.
  cout << problems;
}
