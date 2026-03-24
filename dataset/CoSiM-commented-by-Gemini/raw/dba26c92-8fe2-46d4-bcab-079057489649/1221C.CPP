/**
 * @file 1221C.CPP
 * @brief Solution for Codeforces Problem 1221C - "Perfect Team".
 *
 * @details
 * This program aims to find the maximum number of three-person teams that can be formed
 * from a pool of 'c' coders, 'm' mathematicians, and 'x' without a specialization.
 * A team requires one of each type, but specialists can fill a no-specialization role.
 *
 * The correct and most direct solution is to find the maximum `k` such that:
 * k <= c
 * k <= m
 * k <= (c + m + x) / 3
 * This simplifies to `k = min(c, m, (c + m + x) / 3)`.
 *
 * The implementation below attempts to solve this using a non-standard procedural
 * balancing approach rather than the direct mathematical formula.
 *
 * Time Complexity: O(T) for T test cases, as each test case is solved in O(1).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Fast I/O
  ios_base::sync_with_stdio(false);

  int t;
  cin >> t;

  int c, m, x, smaller, larger;
  int res;

  // Loop through all test cases.
  for (int i = 0; i < t; i++) {
    res = 0;
    cin >> c >> m >> x;

    int maxi = max(c, m);
    int mini = min(c, m);

    // Block Logic: This section appears to try and balance the number of coders and
    // mathematicians by notionally converting the surplus of the larger group
    // into no-specialization members.
    x += (maxi - mini);

    /**
     * Pre-condition: If, after balancing, the pool of no-specialization members is
     * abundant, the number of teams is limited only by the minimum of coders or
     * mathematicians.
     */
    if (x >= c || x >= m) {
      cout << min(c, m) << "
";
      continue;
    }

    // Block Logic: The following steps are an obscure procedural attempt to calculate
    // the remaining teams after the initial balancing and checks. The logic
    // is difficult to interpret and does not follow a standard algorithm.
    maxi = mini;
    maxi -= x;
    mini -= x;

    x += (maxi + mini) / 3;

    cout << x << "
";
  }
}
