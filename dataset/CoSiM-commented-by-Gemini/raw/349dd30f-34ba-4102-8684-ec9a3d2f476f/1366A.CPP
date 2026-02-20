/**
 * @file 1366A.CPP
 * @brief An attempt to solve a resource allocation problem.
 *
 * This program reads 'n' and 'm', representing two types of resources.
 * It iteratively calculates a number of "items" based on a set of rules,
 * likely related to a problem where items cost (1, 2) or (2, 1) of the resources.
 *
 * The code implements a greedy strategy that is complex and may not be correct
 * for all cases of the intended problem (e.g., Shovels and Swords).
 * A correct solution for that problem is typically `min(n, m, (n + m) / 3)`.
 *
 * Time Complexity: O(log(min(n, m))) per test case, due to the resource reduction.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {

  int t;
  cin >> t;
  while (t--) {
    int n, m;
    cin >> n >> m;
    int count = 0;

    /**
     * Block Logic: Greedily consume resources until one is nearly depleted.
     * The loop continues as long as both resource counts are positive.
     */
    while (n > 0 && m > 0) {
      /**
       * Case 1: Resources are highly imbalanced (one is at least twice the other).
       * In this case, the logic assumes that `min(m, n)` items can be made,
       * and fully depletes the scarcer resource.
       */
      if (2 * n <= m || 2 * m <= n) {
        count += min(m, n);
        if (m < n) {
          m = 0;
        } else {
          n = 0;
        }
      } else {
        /**
         * Case 2: Resources are relatively balanced.
         * The logic here is more complex. It calculates a number of items 'k'
         * based on halving the smaller resource. It then reduces both resources
         * based on this 'k' value, simulating a batch crafting process.
         */
        int k = min(m, n) / 2;
        count += k;
        if (m <= n) {
          n -= 2 * (m / 2);
          m -= m / 2;

        } else {
          m -= 2 * (n / 2);
          n -= n / 2;
        }
      }
    }
    cout << count << endl;
  }
}
