/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem from Codeforces.
 * @details This program counts the number of problems a team of three will decide to
 * implement. The condition for implementation is that at least two of the three
 * members are sure about the solution (represented by a '1').
 *
 * @algorithm Explicit Counting
 * For each problem, this solution explicitly counts how many members are confident.
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a final answer counter, `ans`, to zero.
 * 3. Loop `n` times to process each problem.
 * 4. For each problem, initialize a local counter `cnt` to zero.
 * 5. Read the three opinions (0 or 1).
 * 6. Use three separate `if` statements to check each opinion. If an opinion is '1',
 *    increment the local counter `cnt`.
 * 7. After checking all three opinions, if `cnt` is 2 or more, increment the
 *    final answer counter `ans`.
 * 8. After the loop, print the value of `ans`.
 *
 * @complexity
 * - Time: O(N), where N is the number of problems, as there is a single loop
 *   that runs N times with constant-time operations inside.
 * - Space: O(1), for storing a few integer variables.
 */

#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  // n: the total number of problems.
  // ans: the final count of problems the team will solve.
  int n, ans = 0;
  cin >> n;
  
  /**
   * @brief Process each problem one by one.
   */
  while (n--) {
    // a, b, c: store the confidence of the three team members (1 or 0).
    int a, b, c;
    // cnt: counts how many members are sure about the *current* problem.
    int cnt = 0;
    cin >> a >> b >> c;
    
    // Explicitly count the number of members who are sure.
    if (a == 1)
      cnt++;
    if (b == 1)
      cnt++;
    if (c == 1)
      cnt++;
    
    // Pre-condition: Check if at least two members are sure.
    if (cnt >= 2)
      ans++; // If so, increment the total count of solvable problems.
  }
  
  // Print the final answer.
  cout << ans;
  
  return 0; // Indicate successful execution.
}
