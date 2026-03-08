/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 *
 * @details
 * This program determines the number of problems a team of three will solve.
 * A problem is solved if at least two out of the three members are confident
 * about the solution (represented by a 1). The program iterates through all
 * the given problems, counts the number of confident members for each, and if
 * the count is two or more, it increments a total counter for solvable problems.
 *
 * @section
 * Algorithm: Iterative counting using a temporary, resettable counter.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: total number of problems.
  // x, y, z: confidence levels (1 or 0) for the three members.
  // ok: a temporary counter for confident members for the CURRENT problem.
  // yes: the total count of problems that will be solved.
  int n, x, y, z, ok = 0, yes = 0;
  cin >> n;

  /**
   * @brief Main loop to iterate through each of the 'n' problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;

    // Block-Logic: Sum the number of confident members for the current problem.
    if (x == 1)
      ok++;
    if (y == 1)
      ok++;
    if (z == 1)
      ok++;

    // Pre-condition: Check if at least two members are confident.
    if (ok >= 2) {
      // Invariant: If the condition is met, increment the total count of solvable problems.
      yes++;
    }
    // Reset the temporary counter for the next problem. This is critical.
    ok = 0;
  }

  // Output the total number of problems the team will solve.
  cout << yes << endl;
}
