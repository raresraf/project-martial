/**
 * @file 231A.CPP
 * @brief This program determines the number of problems that can be implemented.
 * For each problem, three friends (represented by 'a', 'b', 'c') provide their opinion
 * (1 if sure, 0 if not sure). A problem is implemented if at least two friends
 * are sure about the solution. This implementation uses boolean logic (bitwise AND)
 * for checking the majority condition.
 *
 * Algorithm: Iterative problem evaluation based on a simple majority rule.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1)
 */

#include <bits/stdc++.h> // Includes iostream, etc.
using namespace std;

int main() {
  /**
   * Functional Utility: Optimizes standard input/output operations by unsynchronizing C++ streams
   * from C standard streams and unties cin from cout, improving performance for competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  // Functional Utility: 'n' stores the total number of problems to evaluate.
  short n;
  // Functional Utility: 'a', 'b', 'c' store the opinions of the three friends (Petya, Vasya, Tonya) respectively.
  short a, b, c;
  // Functional Utility: 'cnt' (count) accumulates the number of problems that can be implemented.
  short cnt = 0;

  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to evaluate if it can be implemented.
   * Invariant: At the start of each iteration, 'n' represents the remaining problems to process.
   */
  while (n--) {
    // Block Logic: Reads the opinions of the three friends for the current problem.
    cin >> a >> b >> c;
    /**
     * Block Logic: Checks if at least two friends are sure about the solution using boolean logic.
     * Pre-condition: 'a', 'b', 'c' contain the opinions for the current problem (0 or 1).
     * Invariant: If any two friends are sure (e.g., a AND b is true), 'cnt' is incremented.
     * Inline: `(a && b)` checks if both A and B are sure. The OR conditions ensure that if any
     *         pair is sure, the overall condition is met.
     */
    if ((a && b) || (b && c) || (c && a))
      cnt++;
  }
  // Output the total number of problems that can be implemented.
  cout << cnt << "
";
}
