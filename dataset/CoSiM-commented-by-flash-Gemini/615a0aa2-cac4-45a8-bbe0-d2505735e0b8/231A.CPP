/**
 * @615a0aa2-cac4-45a8-bbe0-d2505735e0b8/231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A). It reads 'n' problems,
 * and for each problem, three binary inputs (0 or 1) indicating whether each of three
 * friends is sure about the solution. A problem is counted if at least two friends are
 * sure about its solution.
 *
 * Algorithm: Two-pass approach: first pass to read inputs into a 2D array, second pass
 * to apply a conditional check based on the sum of pairs of certainty values.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(N) for storing the 2D array of certainty values.
 */

#include <iostream> // Includes the standard input/output stream library.

using namespace std; // Uses the standard namespace.

int main() {
  int n, count = 0; // Declares 'n' for the number of problems and 'count' (solvable problems) initialized to 0.
  cin >> n;         // Reads the number of problems from standard input.
  
  // Declares a 2D array to store the certainty values (0 or 1) for each of the 'n' problems.
  // Each row corresponds to a problem, and columns to friends (e.g., friend 0, friend 1, friend 2).
  int a[n][3]; 
  
  /**
   * Block Logic: First pass - reads the certainty values for each problem into the 2D array 'a'.
   * Invariant: After this nested loop, 'a[i][j]' stores the certainty of friend 'j' for problem 'i'.
   */
  for (int i = 0; i < n; i++) { // Outer loop iterates through each problem.
    for (int j = 0; j < 3; j++) { // Inner loop iterates through the three friends' certainties.
      cin >> a[i][j]; // Reads a certainty value (0 or 1).
    }
  }
  
  /**
   * Block Logic: Second pass - processes each problem to determine if it will be solved.
   * Invariant: 'count' accumulates the total number of problems that meet the criteria.
   */
  for (int i = 0; i < n; i++) { // Loops through each problem.
    /**
     * Block Logic: Checks if at least two friends are sure about the solution for the current problem.
     * Pre-condition: 'a[i][0]', 'a[i][1]', 'a[i][2]' contain binary certainty values.
     * Invariant: 'count' is incremented if any two friends are sure.
     * Inline: The condition `a[i][x] + a[i][y] == 2` effectively checks if both friend 'x' and 'y' are sure (both are 1).
     *         The logical OR (`||`) ensures that if any pair has a sum of 2, the problem is counted.
     */
    if (a[i][0] + a[i][1] == 2 || a[i][1] + a[i][2] == 2 ||
        a[i][2] + a[i][0] == 2) { 
      count += 1; // Increments 'count' if at least two friends are sure.
    } else {
      continue; // Skips to the next problem if the condition is not met.
    }
  }
  cout << count; // Prints the final number of problems the team will solve.
  return 0;      // Indicates successful program termination.
}
