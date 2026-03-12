/**
 * @5f2c4025-4445-4540-8caa-1d6a4bcec8df/231A.CPP
 * @brief This program solves the "Team" problem (Codeforces 231A). It reads 'n' problems,
 * and for each problem, three binary inputs (0 or 1) representing whether each of three
 * friends is sure about the solution. A problem is counted if at least two friends are
 * sure about its solution.
 *
 * Algorithm: Iterative processing of problems, counting confident friends and applying a majority rule.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream.
#include <iostream>      // Explicitly included for iostream.
using namespace std;     // Uses the standard namespace.

int main() {
  int n, ans = 0; // Declares 'n' for the number of problems and 'ans' (solvable problems) initialized to 0.
  cin >> n;       // Reads the number of problems from standard input.
  
  /**
   * Block Logic: Processes 'n' problems. For each problem, it reads the certainty of three friends
   * and increments a counter if at least two friends are sure.
   * Invariant: 'ans' accurately reflects the cumulative number of implemented problems after each iteration.
   */
  while (n--) { // Loops 'n' times, decrementing 'n' in each iteration.
    int a, b, c; // Declares 'a', 'b', 'c' as binary indicators (0 or 1) for each friend's certainty.
    int cnt = 0; // Initializes 'cnt' to 0 for the current problem, to tally friends who are sure.
    cin >> a >> b >> c; // Reads the three binary certainty values (0 or 1) for the current problem.
    
    // Block Logic: Tallies the number of friends who are sure about the solution.
    // Invariant: 'cnt' stores the number of friends (0 to 3) who are sure for the current problem.
    if (a == 1) // Inline: If friend 'a' is sure (a=1), increment cnt.
      cnt++;
    if (b == 1) // Inline: If friend 'b' is sure (b=1), increment cnt.
      cnt++;
    if (c == 1) // Inline: If friend 'c' is sure (c=1), increment cnt.
      cnt++;
    
    /**
     * Block Logic: Checks if the problem meets the implementation criteria.
     * Pre-condition: 'cnt' holds the total number of friends who are sure about the current problem.
     * Invariant: 'ans' is incremented if 'cnt' is 2 or more (at least two friends are sure).
     */
    if (cnt >= 2) // If at least two friends are sure.
      ans++;      // Increment the total count of implemented problems.
  }
  cout << ans; // Prints the final number of problems the friends will implement.
  return 0;    // Indicates successful program termination.
}