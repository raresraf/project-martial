/**
 * @file 1221C.CPP
 * @brief Logic for maximizing resource-constrained team formation.
 * 
 * Logic: Calculates the maximum number of teams possible given 'c' (coders), 
 * 'm' (mathematicians), and 'x' (researchers). Each team requires at least 
 * 1 coder and 1 mathematician, and a total of 3 people.
 * 
 * Formula: Max teams = min(c, m, (c + m + x) / 3).
 * 
 * Algorithm: Piecewise logic to handle different bottleneck scenarios.
 * Time Complexity: $O(T)$ where T is the number of test cases.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  long long int c, m, x, mn, a = 0;
  // Reads total independent scenarios.
  cin >> n;

  /**
   * Block Logic: Evaluates each scenario for team formation capacity.
   * Invariant: Teams are limited by the least abundant required role (coder or mathematician) 
   * or by total headcount.
   */
  while (n--) {
    cin >> c >> m >> x;
    
    // Identifies the minimal count across all categories.
    mn = min(c, min(m, x));

    /**
     * Functional Utility: Edge case handling for impossible team formation.
     */
    if (c == 0 || m == 0)
      cout << 0 << endl;
    
    /**
     * Logic: Small integer case handling (Threshold logic).
     */
    else if ((c == 1 && (c + m + x) >= 3) || (m == 1 && (c + m + x) >= 3) ||
             (c && m && (c + m + x) == 3)) {
      cout << 1 << endl;
    } 
    
    /**
     * Case 3: Complex resource distribution.
     */
    else {
      /**
       * Logic: If either specialized role (c or m) is the global minimum, 
       * it becomes the absolute bottleneck.
       */
      if (mn == m || mn == c) {
        cout << mn << endl;
      } else {
        /**
         * Logic: Researcher count (x) is the minimum.
         * Optimization: Allocates researchers to mn teams, then evaluates 
         * remaining coders and mathematicians for additional teams.
         */
        long long ans = mn;
        c -= mn;
        m -= mn;
        m = max(a, m);
        c = max(a, c);
        
        // Final Capacity: Further teams limited by specialized roles or remaining total count.
        ans += min(min(m, c), ((m + c) / 3));
        cout << ans << endl;
      }
    }
  }
  return 0;
}
