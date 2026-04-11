/**
 * @file 231A.CPP
 * @brief This program determines the number of problems that can be implemented.
 * For each problem, three friends (Petya, Vasya, and Tonya) provide their opinion
 * (1 if sure, 0 if not sure). A problem is implemented if at least two friends
 * are sure about the solution.
 *
 * Algorithm: Iterative problem evaluation based on a simple majority rule.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1)
 */

#include <algorithm> // Not explicitly used but included by competitive programming template.
#include <iostream>
#include <vector>    // Not explicitly used but included by competitive programming template.

using namespace std;

int main(int argc, char const *argv[]) {
  // Functional Utility: 'n' stores the total number of problems to evaluate.
  int n;
  // Functional Utility: 'p' stores Petya's opinion (1 if sure, 0 otherwise).
  int p;
  // Functional Utility: 'v' stores Vasya's opinion (1 if sure, 0 otherwise).
  int v;
  // Functional Utility: 't' stores Tonya's opinion (1 if sure, 0 otherwise).
  int t;
  // Functional Utility: 'ans' counts the number of problems that can be implemented.
  int ans = 0;

  // Block Logic: Reads the total number of problems 'n' from standard input.
  cin >> n;

  /**
   * Block Logic: Iterates through each problem to evaluate if it can be implemented.
   * Invariant: At the start of each iteration, 'i' represents the number of problems already evaluated.
   */
  for (int i = 0; i < n; i++) {
    // Block Logic: Reads the opinions of the three friends for the current problem.
    cin >> p >> v >> t;
    /**
     * Block Logic: Checks if at least two friends are sure about the solution.
     * Pre-condition: 'p', 'v', and 't' contain the opinions for the current problem.
     * Invariant: If the sum of opinions is 2 or more, 'ans' is incremented.
     */
    if (p + v + t >= 2)
      ans++;
  }
  // Output the total number of problems that can be implemented.
  cout << ans;
  return 0;
}
