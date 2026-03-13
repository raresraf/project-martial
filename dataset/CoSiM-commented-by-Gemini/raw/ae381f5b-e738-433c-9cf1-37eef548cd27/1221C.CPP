/**
 * @file 1221C.CPP
 * @brief Solves the "Perfect Team" problem from Codeforces.
 *
 * This program calculates the maximum number of teams that can be formed from
 * `c` coders, `m` mathematicians, and `x` other members. A "perfect team"
 * consists of one coder, one mathematician, and one other person.
 *
 * The maximum number of teams `k` is limited by several constraints:
 * 1. The number of teams cannot exceed the number of coders: `k <= c`.
 * 2. The number of teams cannot exceed the number of mathematicians: `k <= m`.
 * 3. The total number of people required is `3*k`, which cannot exceed the
 *    total number of people available: `3*k <= c + m + x`.
 *
 * Therefore, the maximum number of teams is `min(c, m, (c + m + x) / 3)`.
 * The code implements this logic through a case-based analysis.
 *
 * @see https://codeforces.com/problemset/problem/1221/C
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  long long int c, m, x, mn, a = 0;
  cin >> n;
  while (n--) {
    cin >> c >> m >> x;
    mn = min(c, min(m, x));
    if (c == 0 || m == 0 || (c + m + x) < 0)
      cout << 0 << endl;
    else if ((c == 1 && (c + m + x) >= 3) || (m == 1 && (c + m + x) >= 3) ||
             (c && m && (c + m + x) == 3)) {
      cout << 1 << endl;
    } else {
      /**
       * @brief Main logic for calculating the maximum number of teams.
       *
       * This block calculates the maximum number of teams based on the
       * available members. The logic can be simplified to finding the
       * minimum of `c`, `m`, and `(c + m + x) / 3`.
       */
      if (mn == m || mn == c) {
        // This case handles when the number of coders or mathematicians is the
        // bottleneck. The number of teams is limited by the minimum of these two.
        cout << min(c, m) << endl;
      } else {
        // This case handles when the number of "other" members is initially the
        // smallest group. It forms `mn` (which is `x`) teams first, then
        // calculates how many more teams can be formed from the remaining
        // coders and mathematicians.
        long long ans = mn;
        c -= mn;
        m -= mn;
        m = max(a, m);
        c = max(a, c);
        // The remaining members (c and m) can be grouped to form more teams.
        // Each new team needs 3 members.
        ans += min(min(m, c), ((m + c) / 3));
        cout << ans << endl;
      }
    }
  }
  return 0;
}
