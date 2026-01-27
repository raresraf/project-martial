/**
 * @file 231A.CPP
 * @brief Solution for Codeforces problem 231A "Team".
 *
 * This program determines how many problems a team can solve. A problem is
 * considered solvable if at least two out of three teammates are sure about the solution.
 *
 * Algorithm: Iterative processing of N problems. For each problem, it counts the
 * number of teammates who are sure about the solution and increments a counter if
 * this count is two or more.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(N) to store problem decisions, or O(1) if processed on the fly.
 */
#include <iostream>
using namespace std;
int main() {
  int n, count = 0;
  int a[1000][3];
  // Block Logic: Reads the total number of problems `n`.
  cin >> n;
  /**
   * Block Logic: Reads the decisions of three teammates for each of the `n` problems.
   * A '1' indicates certainty, '0' indicates uncertainty.
   * Pre-condition: `n` is a positive integer representing the number of problems.
   * Invariant: The `a` array stores the decisions for all problems.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  /**
   * Block Logic: Iterates through each problem to determine if it is solvable.
   * A problem is solvable if at least two teammates are sure (flag >= 2).
   * Pre-condition: The `a` array contains problem decisions.
   * Invariant: `count` accumulates the number of solvable problems.
   */
  for (int i = 0; i < n; i++) {
    int flag = 0;
    // Functional Utility: Checks if the first teammate is sure.
    if (a[i][0] == 1)
      flag++;
    // Functional Utility: Checks if the second teammate is sure.
    if (a[i][1] == 1)
      flag++;
    // Functional Utility: Checks if the third teammate is sure.
    if (a[i][2] == 1)
      flag++;
    // Block Logic: Increments the solvable problem count if at least two teammates are sure.
    if (flag >= 2)
      count++;
  }
  // Functional Utility: Prints the total number of solvable problems.
  cout << count;
  return 0;
}
