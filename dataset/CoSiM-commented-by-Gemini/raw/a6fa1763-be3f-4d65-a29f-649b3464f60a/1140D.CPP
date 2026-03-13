/**
 * @file 1140D.CPP
 * @brief Solution for CodeForces Problem 1140D - "Minimum Triangulation".
 * @see https://codeforces.com/problemset/problem/1140/D
 *
 * @details
 * The problem asks for the minimum total weight of a triangulation of a regular n-gon
 * with vertices labeled 1 to n. The weight of a triangle is the product of its vertex labels.
 *
 * @algorithm
 * The minimum triangulation is always achieved by connecting one vertex (say, vertex 1)
 * to all other non-adjacent vertices. This decomposes the n-gon into `n-2` triangles:
 * (1, 2, 3), (1, 3, 4), ..., (1, n-1, n).
 *
 * The total weight is the sum of the weights of these triangles:
 * Weight = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (n-1) * n)
 *
 * The problem statement, however, seems to have been interpreted by the community
 * and solutions as the sum of `i * (i+1)` for `i` from 2 to `n-1`. This is equivalent
 * to the sum of products of the other two vertices in the triangulation from vertex 1,
 * which is: `2*3 + 3*4 + ... + (n-1)*n`.
 *
 * The code implements this summation: `sum_{i=3 to n} i * (i-1)`.
 * Let k = i-1. When i=3, k=2. When i=n, k=n-1.
 * So the sum is equivalent to `sum_{k=2 to n-1} (k+1) * k`.
 *
 * @time_complexity O(N), where N is the number of vertices.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  ll n; // Number of vertices in the polygon.
  ll w = 0; // Total weight of the triangulation.
  
  cin >> n;

  /**
   * @brief Loop to calculate the minimum triangulation weight.
   * This loop calculates the sum of `i * (i-1)` for `i` from 3 to `n`.
   * This corresponds to the sum of products of adjacent vertices in the
   * fan triangulation from a single vertex (e.g., vertex 1).
   * The terms are: 3*2, 4*3, 5*4, ..., n*(n-1).
   */
  for (ll i = 3; i <= n; i++) {
    w = w + (i * (i - 1));
  }
  
  cout << w;
  return 0;
}
