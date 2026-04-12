/**
 * @file 1221C.CPP
 * @brief Logic for maximizing resource allocation in a multi-constrained team formation problem.
 * 
 * Algorithm: Closed-form optimization based on resource bottleneck analysis.
 * Formula: Max teams = min(Coders, Mathematicians, (Coders + Mathematicians + Researchers) / 3).
 * Time Complexity: $O(T)$ where T is the number of test cases.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>
#include <stdio.h>
using namespace std;

#define ll long long
#define scl(n) scanf("%lld", &n)
#define fr(i, n) for (ll i = 0; i < n; i++)
#define fr1(i, n) for (ll i = 1; i <= n; i++)
#define pfl(x) printf("%lld\n", x)
#define endl "\n"
#define pb push_back
#define l(s) s.size()
#define asort(a) sort(a, a + n)
#define dsort(a) sort(a, a + n, greater<int>())
#define vasort(v) sort(v.begin(), v.end());
#define vdsort(v) sort(v.begin(), v.end(), greater<int>());
#define pn printf("\n")
#define md 10000007

int main() {
  ll m, n, t, b, c, d, i, j, k, x, y, z, l, q, r;
  ll cnt = 0, ans = 0;
  
  // Reads the number of independent test cases.
  scl(t);
  
  /**
   * Block Logic: Evaluates team formation potential for each test case.
   * Invariant: A team must have exactly one coder, one mathematician, and one 
   * additional person from any category.
   */
  while (t--) {
    // x: Coders, y: Mathematicians, z: Researchers.
    cin >> x >> y >> z;
    
    /**
     * Logic: The primary bottleneck is the number of available coders and mathematicians.
     * A coder and a mathematician are strictly required for each team.
     */
    cnt = min(x, y);

    /**
     * Functional Utility: Resource balance calculation.
     * Case 1: Total required specialists (x+y) are sufficient even when supplemented by researchers (z).
     * Case 2: The total headcount (x+y+z) is the ultimate limiting factor.
     */
    if (cnt <= z)
      ans = cnt;
    else {
      // Logic: Maximum teams cannot exceed 1/3 of the total available human resources.
      ans = (x + y + z) / 3;
      // Constraint: Even with surplus people, we cannot exceed the specialist bottleneck.
      if (ans > cnt)
        ans = cnt;
    }
    
    // Output the optimized team count for the current scenario.
    pfl(ans);
  }
  return 0;
}
