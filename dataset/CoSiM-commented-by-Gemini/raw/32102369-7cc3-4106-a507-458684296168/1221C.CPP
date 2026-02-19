/**
 * @file 1221C.CPP
 * @brief Solves a team formation optimization problem using binary search on the answer.
 * @details The problem is to find the maximum number of 3-person teams that can be formed
 * from `c` coders, `m` mathematicians, and `x` other specialists. Each team must have
 * one coder, one mathematician, and one other person. The "other" person can be a
 * remaining coder, a remaining mathematician, or a specialist.
 *
 * @algorithm The solution uses "Binary Search on the Answer". The maximum number of teams `k` must satisfy:
 * 1. `k <= c` (enough coders)
 * 2. `k <= m` (enough mathematicians)
 * 3. `k <= (c - k) + (m - k) + x` which simplifies to `3k <= c + m + x` (enough people for the third slot in each team).
 *
 * The code binary searches for the largest `k` (named `mid`) in the range `[0, min(c, m)]`, which already
 * satisfies the first two conditions. For each `mid`, it checks the third condition. If it holds,
 * `mid` is a possible answer, and a larger `k` is explored. Otherwise, `k` is too large.
 *
 * @time_complexity O(Q * log(min(c, m))), where Q is the number of queries.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, j, q, c, m, x, ans;
  cin >> q;
  /**
   * Block Logic: Process each of the 'q' queries independently.
   */
  while (q--) {
    cin >> c >> m >> x;
    // The search space for the number of teams is from 0 to min(c, m), as we
    // cannot form more teams than we have coders or mathematicians.
    int l = 0, r = min(c, m);
    ans = 0; // Initialize answer for the current query.
    /**
     * Block Logic: Binary search for the maximum possible number of teams (`ans`).
     * `mid` represents a potential number of teams to form.
     */
    while (l <= r) {
      int mid = (l + r) / 2;
      // Pre-condition: Check if there are enough people in total to form `mid` teams of 3.
      // This is equivalent to checking if `c + m + x >= 3 * mid`.
      // The `c + m + x - 2 * mid` represents the pool of people left for the 3rd slot
      // after `mid` coders and `mid` mathematicians have been assigned to teams.
      if (c + m + x - 2 * mid >= mid) {
        // `mid` is a valid number of teams. Store it and try for more.
        ans = mid;
        l = mid + 1;
      } else {
        // `mid` is too high, not enough people. Reduce the search space.
        r = mid - 1;
      }
    }
    cout << ans << endl;
  }
}
