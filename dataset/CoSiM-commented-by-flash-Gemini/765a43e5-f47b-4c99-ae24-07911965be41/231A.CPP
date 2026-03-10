/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A).
 *   It counts the number of problems for which at least two out of three participants are sure about the solution.
 *   Problem description: https://codeforces.com/problemset/problem/231/A
 * * Algorithm: Iterative counting. For each problem, it sums the certainty votes (0 or 1) from three participants.
 *   If the sum is 2 or more, it increments a counter.
 * Time Complexity: O(N) where N is the number of problems, as each problem is processed in constant time.
 * Space Complexity: O(1) - constant space, as only a few variables are used.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the problem counting logic.
 * Functional Utility: Reads the number of problems, then processes each problem's certainty votes,
 * and outputs the total count of problems where at least two participants are sure.
 */
int main() {
  int n, a, b, c, r;
  // Block Logic: Reads the number of problems 'n'.
  cin >> n;
  // Initialize 'r' to count problems with at least two sure participants.
  r = 0;
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   * Pre-condition: 'n' is a non-negative integer representing the total number of problems.
   * Invariant: In each iteration, 'a', 'b', 'c' represent the certainty (1 if sure, 0 if not) of three participants for the current problem.
   */
  for (int i = 0; i < n; i++) {
    // Reads the certainty votes from three participants for the current problem.
    cin >> a >> b >> c;
    /**
     * Block Logic: Checks if at least two participants are sure about the solution for the current problem.
     * If the sum of their votes is 2 or greater, the problem is considered solved.
     */
    if (a + b + c >= 2)
      ++r; // Increments the counter if the condition is met.
  }
  // Outputs the total number of problems for which at least two participants were sure.
  cout << r << endl;
  return 0;
}
