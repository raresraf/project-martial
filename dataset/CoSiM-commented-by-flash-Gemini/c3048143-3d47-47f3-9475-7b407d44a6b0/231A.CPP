/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem. It counts the number of problems
 * a team will solve, given their confidence levels. For each problem, three friends
 * state their confidence (0 or 1). A problem is solved if at least two friends are
 * sure (confidence level of 1).
 * Algorithm: Explicit conditional checking for each combination of confident friends.
 * Time Complexity: O(N), where N is the number of problems, due to a single pass.
 * Space Complexity: O(1) as only a few variables are used per iteration.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.
#include <string.h>      // C string manipulation library, not explicitly used in this solution.
using namespace std;

int main() {
  // Optimizes C++ standard streams for faster input/output operations.
  ios::sync_with_stdio(0);
  cin.tie(0);  // Unties cin from cout.
  cout.tie(0); // Unties cout.

  int n;      // Stores the total number of problems to be considered.
  int counts = 0; // Counts the number of problems the team will solve.
  int petya, vasaya, tonya; // Stores the confidence levels of the three friends.

  // Reads the total number of problems 'n'.
  cin >> n;
  /**
   * Block Logic: Processes each of the 'n' problems.
   * For each problem, it reads the confidence levels of three friends and determines
   * if the problem is solved based on whether at least two friends are confident.
   * Invariant: 'counts' correctly reflects the number of solved problems up to the current iteration.
   */
  for (int i = 0; i < n; i++) {
    // Reads the confidence levels of Petya, Vasya, and Tonya for the current problem.
    cin >> petya;
    cin >> vasaya;
    cin >> tonya;
    /**
     * Conditional Logic: Checks if at least two out of three friends are confident (have a value of 1).
     * This is most efficiently determined by summing their confidence levels; if the sum is 2 or more,
     * then at least two friends are confident, and the problem is considered solved.
     */
    if ((petya == 1 && vasaya == 1 && tonya == 1) || // All three are confident.
        (petya == 1 && vasaya == 1 && tonya == 0) || // Petya and Vasya are confident.
        (petya == 1 && vasaya == 0 && tonya == 1) || // Petya and Tonya are confident.
        (petya == 0 && vasaya == 1 && tonya == 1)) { // Vasya and Tonya are confident.
      counts = counts + 1; // Increments the solved problems counter.
    } else {
      counts = counts + 0; // If not solved, counter remains unchanged (explicitly adds 0).
    }
  }
  // Prints the total number of problems the team will solve.
  cout << counts;

  // Standard program exit.
  return 0;
}
