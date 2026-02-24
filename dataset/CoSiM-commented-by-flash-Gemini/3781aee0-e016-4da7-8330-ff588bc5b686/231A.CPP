/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - Team.
 *
 * This program determines the number of problems a team can solve in a contest.
 * A problem is considered solved if at least two out of three team members
 * (Petya, Vasya, and Tonya) are confident in its solution.
 *
 * Algorithm: Direct conditional check. For each problem, read the confidence
 * level of the three team members and check if at least two of them are confident.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1), as only a few variables are stored.
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * @brief Main function to execute the Team problem logic.
   *
   * Reads the number of problems, then for each problem, reads the confidence
   * votes of three team members. It counts and prints the total number of
   * problems where at least two members are confident.
   */
  // Declare 'n' to store the number of problems.
  int n;
  // Read the total number of problems.
  cin >> n;
  // Declare 'a', 'b', 'c' for the confidence votes of the three team members,
  // and 'ans' to store the count of solved problems.
  int a, b, c, ans = 0;
  // Block Logic: Iterate through each problem.
  // Invariant: 'n' decreases with each iteration, processing one problem per loop.
  while (n--) {
    // Read the confidence votes (1 for confident, 0 for not confident) for the current problem.
    cin >> a >> b >> c;
    // Check if at least two of the three team members are confident (vote == 1).
    // This condition covers all pairs: (a and b) or (a and c) or (c and b) are both 1.
    if ((a == b && a == 1) || (a == c && a == 1) || (c == b && c == 1))
      ans++; // Increment the count of solved problems.
  }
  // Output the total number of problems solved.
  cout << ans;
  return 0;
}