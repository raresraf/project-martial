/**
 * @file 1221C.CPP
 * @brief Solution to Codeforces problem 1221C - Perfect Team.
 *        This program calculates the maximum number of "perfect teams" that can be formed
 *        given a count of coders (c), mathematicians (m), and universal specialists (x).
 *        A perfect team requires one coder, one mathematician, and one additional specialist
 *        (which can be a coder, mathematician, or universal specialist).
 * Algorithm:
 *   1. The core idea is to first balance the number of coders (c) and mathematicians (m)
 *      by moving the excess from the larger group into the universal specialists (x).
 *      This simplifies the problem to a state where c == m.
 *   2. After balancing, all remaining 'c' and 'm' can form c-m-x teams.
 *      The number of teams formed is then limited by the minimum of the three counts (c, m, x),
 *      and also by the total number of specialists divided by 3 (since each team needs 3 members).
 *   3. Specifically, let 'ans' be the number of teams.
 *      Initially, 'ans' is the minimum of c, m, and x (this represents forming c:m:x teams where c=m=x).
 *      The remaining c and m (after forming `ans` teams) are then considered. The total remaining
 *      specialists (c + m + x - 3*ans) can form additional teams, but each team still needs 3 members.
 *      The maximum number of additional teams from the remaining c and m is (c + m) / 3.
 *      So, the final answer is `ans + (c + m) / 3` after these adjustments.
 * Time Complexity: O(1) as it involves a fixed number of arithmetic operations per test case.
 * Space Complexity: O(1) for storing a few variables.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function to solve the "Perfect Team" problem.
   *
   * Reads the number of test cases and for each test case, reads the counts of
   * coders, mathematicians, and universal specialists. It then calculates and
   * prints the maximum number of perfect teams that can be formed.
   * @return int Returns 0 upon successful execution.
   */
  int q, c, m, x;
  cin >> q; // Read the number of queries (test cases).
  // Block Logic: Iterates through each query (test case).
  // Pre-condition: 'q' specifies the total number of test cases.
  // Invariant: 'ans' is reset for each test case to store the maximum teams.
  for (int i = 0; i < q; i++) {
    int ans = 0; // Initialize answer for the current test case.
    cin >> c >> m >> x; // Read counts of coders, mathematicians, and universal specialists.
    
    // Block Logic: Balances the number of coders and mathematicians by moving the excess
    // to universal specialists. This ensures that 'c' and 'm' become equal,
    // maximizing the potential for forming initial 1:1:1 teams.
    // Pre-condition: c, m, x are positive integer counts.
    // Post-condition: min(c, m) is now the new c and m, with the difference added to x.
    if (c > m) {
      x += c - m; // Add excess coders to universal.
      c = m;      // Set coder count equal to mathematician count.
    } else {
      x += m - c; // Add excess mathematicians to universal.
      m = c;      // Set mathematician count equal to coder count.
    }
    
    // Block Logic: Calculates the maximum number of teams.
    // The number of teams is limited by either the minimum of the three categories (c, m, x)
    // or by the total remaining people divided by 3 (since each team needs 3 members).
    // The minimum of c and m here represents the number of pairs of (coder, mathematician).
    // The total people (c + m + x) divided by 3 represents the absolute maximum possible teams.
    // The final answer is the minimum of these two limiting factors.
    ans = min(c, min(m, x));
    c -= ans; // Deduct resources used for initial 'ans' teams.
    m -= ans; // Deduct resources used for initial 'ans' teams.
    ans += (c + m) / 3; // Add additional teams formed from remaining c and m.
    cout << ans << endl; // Output the maximum number of perfect teams for the current query.
  }

  return 0;
}