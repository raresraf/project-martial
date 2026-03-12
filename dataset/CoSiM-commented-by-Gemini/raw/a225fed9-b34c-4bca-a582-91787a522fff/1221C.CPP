//! Bismillahi-Rahamanirahim.
/** ========================================**
 ** @Author: Md. Abu Farhad ( RUET, CSE'15)
 ** @Category:
/** ========================================**/

/**
 * @file 1221C.CPP
 * @brief Solution to the "Perfect Team" problem (Codeforces 1221C).
 * @details The problem asks for the maximum number of teams that can be formed from
 *          `c` coders, `m` mathematicians, and `x` versatile individuals. Each team
 *          must consist of one coder, one mathematician, and one other person, who
 *          can be a coder, a mathematician, or a versatile person.
 *
 * @algorithm Let `k` be the number of teams. To form `k` teams, we need:
 *            1. `k` coders, so `k <= c`.
 *            2. `k` mathematicians, so `k <= m`.
 *            3. `k` other members. These can be drawn from the remaining pool of
 *               `(c - k)` coders, `(m - k)` mathematicians, and `x` versatile people.
 *               So, `k` must be less than or equal to the total number of people available
 *               after forming the core of the teams: `k <= (c - k) + (m - k) + x`.
 *               This simplifies to `3k <= c + m + x`, or `k <= (c + m + x) / 3`.
 *
 *            Combining these constraints, the maximum number of teams `k` is limited by
 *            the minimum of these three values: `k = min(c, m, (c + m + x) / 3)`.
 *            The code implements this formula.
 *
 * @time_complexity O(1) for each test case.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
#include <stdio.h>
using namespace std;

#define ll long long
#define scl(n) scanf("%lld", &n) // Macro for scanning a long long.
#define fr(i, n) for (ll i = 0; i < n; i++)
#define fr1(i, n) for (ll i = 1; i <= n; i++)
#define pfl(x) printf("%lld\n", x) // Macro for printing a long long.
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
  scl(t); // Read the number of test cases.
  while (t--) {
    cin >> x >> y >> z; // Read c, m, x for the current test case.
    
    // The number of teams is at most the minimum of coders and mathematicians.
    cnt = min(x, y);
    
    // The logic below calculates min(x, y, (x+y+z)/3) in a non-obvious way.
    if (cnt <= z) {
        // If min(c, m) <= x, it can be mathematically shown that the overall
        // bottleneck is min(c, m). So, the answer is simply min(c, m).
        ans = cnt;
    }
    else {
        // If min(c, m) > x, the bottleneck could be either min(c, m) or the
        // total number of people. So we must take the minimum of both.
        ans = (x + y + z) / 3;
        if (ans > cnt)
            ans = cnt;
    }
    pfl(ans);
  }
  return 0;
}
