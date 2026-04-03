/**
 * @8b30dc76-370d-4fc4-953b-72fd4160bbf1/231A.CPP
 * @brief This program determines the number of problems a team will implement
 * in a programming contest. For each problem, three team members express their
 * confidence (1 if confident, 0 if not). A problem is implemented if at least
 * two team members are confident.
 * Algorithm: Iterative counting with conditional logic.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) for storing counters and problem confidence levels.
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming
using namespace std;

int main() {
  // @var n: Total number of problems.
  // @var x, y, z: Confidence levels (1 or 0) of the three team members for a problem.
  // @var ok: Counter for confident team members for the current problem.
  // @var yes: Counter for problems that will be implemented.
  int n, x, y, z, ok = 0, yes = 0;
  
  // Block Logic: Read the total number of problems.
  cin >> n;
  
  // Block Logic: Iterate through each problem to determine if it will be implemented.
  // Invariant: 'yes' counts the total implemented problems, 'ok' is reset for each problem.
  // Pre-condition: 'n' is the total number of problems to evaluate.
  for (int i = 0; i < n; i++) {
    // Block Logic: Read confidence levels for the current problem.
    cin >> x >> y >> z;
    
    // Inline: Increment 'ok' if the first team member is confident.
    if (x == 1)
      ok++;
    // Inline: Increment 'ok' if the second team member is confident.
    if (y == 1)
      ok++;
    // Inline: Increment 'ok' if the third team member is confident.
    if (z == 1)
      ok++;
    
    // Block Logic: Check if at least two team members are confident.
    // If true, increment the count of implemented problems.
    if (ok >= 2) {
      yes++;
    }
    // Inline: Reset 'ok' counter for the next problem.
    ok = 0;
  }
  
  // Post-condition: Print the total number of problems that will be implemented.
  cout << yes << endl;
  return 0; // Explicitly return 0 for successful execution.
}
