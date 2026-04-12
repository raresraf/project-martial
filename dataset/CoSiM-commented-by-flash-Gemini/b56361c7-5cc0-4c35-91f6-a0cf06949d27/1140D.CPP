/**
 * @file 1140D.CPP
 * @brief Greedy algorithm for calculating the minimum weight triangulation of a regular polygon.
 * 
 * Logic: For a convex polygon with n vertices labeled 1..n, the weight of a triangle 
 * (i, j, k) is i*j*k. The optimal triangulation (minimum total weight) is achieved 
 * by using vertex 1 as a common pivot for all (n-2) triangles in the form (1, i, i+1).
 * 
 * Algorithm: Closed-form greedy summation.
 * Formula: Total Weight = $\sum_{i=2}^{n-1} (1 \times i \times (i+1))$.
 * 
 * Time Complexity: $O(N)$ where N is the number of vertices.
 * Space Complexity: $O(1)$ constant auxiliary space.
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

using namespace std;

typedef long long ll;

int main() {
  int n;
  // Reads the total number of vertices in the polygon.
  if (!(cin >> n)) return 0;
  
  ll total_weight = 0LL;

  /**
   * Block Logic: Summation of individual triangle weights.
   * Invariant: Each iteration adds the weight of triangle (1, i, i+1).
   * Range: i traverses from 2 up to n-1.
   */
  for (int i = 2; i < n; ++i) {
    /**
     * Functional Utility: Computes weight of triangle with vertices {1, i, i+1}.
     * Logic: Since 1 is the multiplicative identity, weight simplifies to i * (i+1).
     */
    total_weight += (ll)i * (ll)(i + 1);
  }
  
  // Output the minimum possible sum of triangle weights.
  cout << total_weight << endl;
  
  return 0;
}
