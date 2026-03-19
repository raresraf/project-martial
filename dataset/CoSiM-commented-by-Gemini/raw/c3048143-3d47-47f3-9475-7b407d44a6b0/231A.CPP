/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @details This program counts the number of problems a team of three will solve,
 * based on the rule that a problem is solved if at least two members are
 * confident in the solution.
 *
 * Algorithm: Explicit Case Enumeration
 * The program iterates through each problem and reads the confidence level
 * (1 for sure, 0 for not sure) of the three team members. It then uses a
 * large conditional statement to explicitly check for every possible combination
 * where at least two members are sure (e.g., {1,1,0}, {1,0,1}, {0,1,1}, {1,1,1}).
 * While logically correct, this approach is more verbose than summing the inputs
 * and checking if the sum is >= 2.
 *
 * Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */

#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  // Optimizes C++ stream I/O for faster execution.
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  
  // n: The total number of problems.
  // counts: A counter for the problems that will be implemented.
  int n, counts = 0;
  // petya, vasaya, tonya: Confidence levels (1 or 0) for each member.
  int petya, vasaya, tonya;
  
  cin >> n;
  
  /**
   * Block Logic: Iterate through each problem to determine if it will be solved.
   * Pre-condition: 'n' holds the number of problems to process.
   */
  for (int i = 0; i < n; i++) {
    cin >> petya;
    cin >> vasaya;
    cin >> tonya;
    
    /**
     * Core Logic: Explicitly check all combinations where at least two members agree.
     * This if-statement covers the four cases:
     * - All three are sure.
     * - Petya and Vasya are sure.
     * - Petya and Tonya are sure.
     * - Vasya and Tonya are sure.
     */
    if ((petya == 1 && vasaya == 1 && tonya == 1) ||
        (petya == 1 && vasaya == 1 && tonya == 0) ||
        (petya == 1 && vasaya == 0 && tonya == 1) ||
        (petya == 0 && vasaya == 1 && tonya == 1)) {
      counts = counts + 1;
    } else {
      // This else block is redundant as adding 0 does not change the value.
      counts = counts + 0;
    }
  }
  
  // Output the final count of problems to be implemented.
  cout << counts;

  return 0;
}