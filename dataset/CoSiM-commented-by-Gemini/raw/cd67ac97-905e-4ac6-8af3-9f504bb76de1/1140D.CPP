/**
 * @file 1140D.CPP
 * @brief Solution for "Minimum Triangulation" (Codeforces 1140D).
 *
 * @details
 * The problem asks for the minimum cost to triangulate a regular n-sided polygon
 * with vertices labeled 1 to n. The cost of a triangulation is the sum of the
 * costs of its constituent triangles, where a triangle's cost is the product
 * of its vertex labels.
 *
 * @section algorithm Algorithm
 * To minimize the total cost, we should aim to minimize the product for each
 * triangle. The optimal strategy is to always include the vertex with the smallest
 * label (vertex 1) in every triangle.
 *
 * This can be achieved by creating a "fan" triangulation from vertex 1. We form
 * n-2 triangles by connecting vertex 1 to all other non-adjacent vertices. The
 * triangles formed are (1, 2, 3), (1, 3, 4), (1, 4, 5), ..., (1, n-1, n).
 *
 * The cost of this triangulation is the sum of the products of the vertex labels
 * for each of these triangles:
 * Cost = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (n-1) * n)
 * This simplifies to the summation:  Sum_{i=2}^{n-1} i * (i+1)
 *
 * The code directly implements this summation.
 *
 * @section complexity Complexity
 * - Time Complexity: O(N), where N is the number of vertices, for the loop.
 * - Space Complexity: O(1).
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

using namespace std;

// The commented-out includes are for policy-based data structures, a common
// feature in competitive programming templates, but not used in this solution.
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
// typedef tree<int,null_type,less<int>,rb_tree_tag,
// tree_order_statistics_node_update> ordered_set;

int main(int argc, char const *argv[]) {
  int a; // The number of vertices of the polygon (n).
  cin >> a;
  long long ans = 0; // Use long long to prevent overflow for large n.

  /**
   * @brief This loop calculates the sum: Sum_{i=2}^{a-1} i * (i+1).
   * This corresponds to the minimum triangulation cost explained above.
   */
  for (int i = 2; i < a; ++i) {
    ans += i * (i + 1);
  }
  cout << ans << endl;
  return 0;
}
