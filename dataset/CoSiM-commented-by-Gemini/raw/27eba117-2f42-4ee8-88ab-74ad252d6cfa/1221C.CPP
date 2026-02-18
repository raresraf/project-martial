// Problem Link : https://codeforces.com/problemset/problem/1221/C
/**
 * @file 1221C.CPP
 * @brief Solution to the "Perfect Team" problem (1221C) from Codeforces.
 * @details The goal is to find the maximum number of teams that can be formed from
 * `c` coders, `m` mathematicians, and `x` universal specialists. Each team requires
 * one coder, one mathematician, and one person from any of the three roles.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The maximum number of teams, `k`, is constrained by three factors:
 * 1. The number of coders: `k <= c`
 * 2. The number of mathematicians: `k <= m`
 * 3. The total number of people: A team needs 3 people, so `3*k <= c + m + x`.
 *
 * The solution is the maximum `k` that satisfies all three, which simplifies to `k = min(c, m, (c+m+x)/3)`.
 * The code implements this logic through a non-obvious case analysis.
 *
 * @complexity
 * Time Complexity: O(1) for each test case, as it involves a few arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
  ll q;
  cin >> q;
  while (q--) {
    ll c, m, x, d; // The variable `d` is declared but never used.
    cin >> c >> m >> x;
    /**
     * @block
     * @brief This block is a heuristic that correctly identifies cases where the
     * number of coders or mathematicians is the primary bottleneck. In such cases,
     * the maximum number of teams cannot exceed `min(c, m)`.
     */
    if (c <= x || m <= x || c == 0 || m == 0) {
      cout << min(c, m) << endl;
    } else {
      /**
       * @block
       * @brief This block handles the general case and correctly calculates `min(c, m, (c+m+x)/3)`.
       *
       * @logic
       * 1. It conceptually uses the `x` specialists to form `x` teams, leaving `c-x` and `m-x` of the
       *    other roles.
       * 2. It then calculates the maximum number of *additional* teams that can be formed from the
       *    remaining `c-x` and `m-x` people. This is limited by `min(c-x, m-x, (c-x + m-x)/3)`.
       * 3. The code's `if/else` structure is a convoluted way of finding this minimum.
       * 4. The final result `x + min(...)` correctly simplifies to the overall `min(c, m, (c+m+x)/3)`.
       */
      c = c - x;
      m = m - x;
      if ((c + m) / 3 > min(c, m))
        x = x + min(c, m);
      else
        x = x + (c + m) / 3;
      cout << x << endl;
    }
  }
  return 0;
}
