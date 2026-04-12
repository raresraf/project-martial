/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem.
 *
 * This program determines the number of problems a team of three friends (Petya, Vasya, Tonya)
 * will implement in a programming contest. A problem is implemented if at least two of the
 * three friends are sure about its solution.
 *
 * Algorithm: Iterative counting based on a sum threshold.
 * Time Complexity: O(N) where N is the number of problems, as each problem's input is processed once.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

/**
 * @brief Main function to calculate the number of problems the team will implement.
 *
 * Functional Utility: Reads the number of problems, then for each problem, reads
 * the confidence levels of the three friends, and increments a counter if at
 * least two friends are sure.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output,
  // which is common in competitive programming.
  _ int n; // Declare 'n' for the number of problems.
  cin >> n; // Read the total number of problems.
  int ans = 0; // Initialize a counter for problems to be implemented.

  /**
   * Block Logic: Processes each problem by reading the confidence levels of the three friends.
   * If the sum of confidence levels (sure votes) is greater than 1 (i.e., at least 2),
   * the problem is considered solvable by the team, and the 'ans' counter is incremented.
   * Pre-condition: 'n' has been read, specifying the total number of problems.
   * Invariant: 'ans' accurately reflects the count of problems implementable by the team so far.
   */
  while (n--) {
    int x = 0, a, b, c; // 'a', 'b', 'c' for friends' confidence, 'x' for their sum.
    cin >> a >> b >> c; // Read confidence levels for the current problem.
    x = a + b + c;      // Calculate the sum of confidence levels.
    if (x > 1)          // Check if at least two friends are sure.
      ans++;            // Increment problem count if solvable.
  }
  cout << ans << "\n"; // Output the total number of implementable problems.

  return 0;
}