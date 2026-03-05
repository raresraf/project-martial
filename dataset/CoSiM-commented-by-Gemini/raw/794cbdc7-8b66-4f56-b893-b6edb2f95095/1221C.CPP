/**
 * @file 1221C.CPP
 * @brief Solution for Codeforces problem 1221C, "Perfect Team".
 * @details The problem is to find the maximum number of teams that can be formed from a pool of `c` coders, `m` mathematicians, and `x` participants with no specialization. A team must consist of one coder, one mathematician, and one other participant. The "other" participant can be a coder, a mathematician, or a no-specialization person.
 *
 * Algorithm:
 * The maximum number of teams, `k`, must satisfy three conditions:
 * 1. The number of teams cannot exceed the number of coders: `k <= c`
 * 2. The number of teams cannot exceed the number of mathematicians: `k <= m`
 * 3. The total number of people required is `3k`. This cannot exceed the total number of people available: `3k <= c + m + x`.
 *
 * Combining these, the maximum number of teams `k` is `min(c, m, (c + m + x) / 3)`.
 * The implementation in this file reaches this result through a series of non-obvious transformations.
 *
 * Time Complexity: O(1) per query.
 * Space Complexity: O(1).
 */
/*
        O(1) solution
        make c and m equal and then move the rest to x;
        make team with 1c 1m 1x and make the rest into 2c1m or 2m1c
        we can make A 2c1m teams and B 2m1c teams
        so A+B = (c+m)/3
        add up the 1c1m1x team with (c+m)/3 is the answer
*/
#include <bits/stdc++.h>

using namespace std;

int main() {
  int q, c, m, x;
  cin >> q;
  /**
   * @brief This loop processes each query independently.
   */
  for (int i = 0; i < q; i++) {
    int ans = 0;
    cin >> c >> m >> x;
    /**
     * @brief The following block of code, though implemented in a confusing way,
     * effectively calculates `min(c, m, (c + m + x) / 3)`. It is not a
     * direct simulation of team formation but a mathematical shortcut.
     */
    if (c > m) {
      x += c - m;
      c = m;
    } else {
      x += m - c;
      m = c;
    }
    ans = min(c, min(m, x));
    c -= ans;
    m -= ans;
    ans += (c + m) / 3;
    cout << ans << endl;
  }

  return 0;
}