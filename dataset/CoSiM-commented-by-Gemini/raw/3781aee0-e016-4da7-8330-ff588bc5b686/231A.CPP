/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem by checking for pairs of '1's.
 *
 * This program determines the number of problems a team of three will solve.
 * A problem is solved if at least two members are sure (represented by a 1).
 * This implementation reads each problem's data and immediately processes it.
 *
 * Algorithm:
 * 1. Read the number of problems, n.
 * 2. Initialize a counter 'ans' to 0.
 * 3. Loop 'n' times:
 *    a. Read the three opinions (a, b, c).
 *    b. Check if any pair of opinions is (1, 1). This implicitly covers the
 *       case where all three are 1s.
 *    c. If the condition is met, increment 'ans'.
 * 4. Print the final count 'ans'.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int n; // Number of problems.
  cin >> n;
  
  // a, b, c: opinions of the three team members.
  // ans: total count of problems that will be solved.
  int a, b, c, ans = 0;
  
  /**
   * Block Logic: Main processing loop.
   * Iterates 'n' times, processing one problem per iteration.
   */
  while (n--) {
    cin >> a >> b >> c;
    
    /**
     * Block Logic: Check for consensus.
     * This condition checks if any two of the three members are sure.
     * For example, `a == b && a == 1` checks if both 'a' and 'b' are 1.
     * This correctly identifies cases where the sum of votes is >= 2.
     */
    if ((a == b && a == 1) || (a == c && a == 1) || (c == b && c == 1))
      ans++;
  }
  
  // Print the final count.
  cout << ans;
  return 0;
}
