/**
 * @file 1140D.CPP
 * @brief Solves the "Minimum Triangulation" problem (Codeforces 1140D).
 *
 * @details
 * The problem asks for the minimum possible total weight of a triangulation of a
 * regular N-gon. The vertices of the polygon are labeled 1 to N in order.
 * The weight of a triangle is the product of the labels of its vertices.
 * A triangulation divides the N-gon into N-2 triangles.
 *
 * Algorithm:
 * To minimize the sum of products (weights), we should always use the smallest
 * possible vertex labels in our triangles. The smallest label is 1.
 * The optimal strategy is to create a "fan" triangulation by picking vertex 1
 * and drawing diagonals to all other non-adjacent vertices (3, 4, ..., N-1).
 * This creates N-2 triangles with vertices:
 * (1, 2, 3), (1, 3, 4), (1, 4, 5), ..., (1, N-1, N).
 *
 * The total weight is the sum of the weights of these triangles:
 * Weight = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (N-1) * N)
 * This simplifies to the sum of `i * (i+1)` for `i` from 2 to N-1.
 * The code implements this summation directly.
 *
 * Time Complexity: O(N), for the loop that calculates the sum.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  long long ans = 0;
  /**
   * @brief Calculates the minimum triangulation weight.
   * This loop sums the products `i * (i+1)` for i from 2 to n-1, which
   * corresponds to the sum of weights of triangles (1, i, i+1) in a fan
   * triangulation from vertex 1.
   */
  for (int i = 2; i <= n - 1; i++) {
    ans += (i * (i + 1));
  }
  cout << ans << "\n";
}