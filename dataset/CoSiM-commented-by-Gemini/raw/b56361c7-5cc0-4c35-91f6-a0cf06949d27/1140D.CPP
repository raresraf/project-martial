/**
 * @file 1140D.CPP
 * @brief Solution for a computational geometry problem, likely related to polygon triangulation.
 *
 * @details
 * This program calculates the sum of `i * (i + 1)` for `i` from 2 to `n-1`. This specific
 * summation corresponds to the cost of a particular triangulation of a convex n-gon,
 * specifically a fan triangulation from a single vertex (e.g., vertex 1). In such a
 * triangulation, the triangles formed are (1, 2, 3), (1, 3, 4), ..., (1, n-1, n).
 * If the cost of a triangle is the product of its vertex labels, the total cost is
 * the sum of these products: `1*2*3 + 1*3*4 + ... + 1*(n-1)*n`. This simplifies to
 * the formula implemented in the code.
 *
 * This approach avoids a more complex dynamic programming solution, suggesting that the
 * problem asks for a minimal triangulation cost, which for this cost function is
 * always the fan triangulation.
 *
 * Algorithm: Iterative summation.
 * Time Complexity: O(n), where n is the number of vertices.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define eps 1e-7
#define M 20005
#define N 150
#define SZ 200000002
using namespace std;
typedef long long ll;
const int inf = 1000000000;
ll mod = 1000000007LL;
ll mod2 = 998244353LL;
int n, k;
int main() {
  cin >> n;
  ll ans = 0LL;
  /**
   * @brief Block Logic: This loop calculates the sum of `i * (i + 1)` for `i` from 2 to `n-1`.
   * @details This corresponds to the cost of a fan triangulation of a convex n-gon,
   * where triangles (1, 2, 3), (1, 3, 4), ..., (1, n-1, n) are formed. The cost
   * of each triangle (1, i, i+1) is `1 * i * (i+1)`. The loop sums these costs.
   *
   * @invariant At the beginning of each iteration `i`, `ans` holds the sum of costs
   * for all triangles up to vertex `i-1`.
   */
  for (int i = 2; i < n; ++i) {
    ans += (ll)i * (ll)(i + 1);
  }
  cout << ans << endl;
  return 0;
}