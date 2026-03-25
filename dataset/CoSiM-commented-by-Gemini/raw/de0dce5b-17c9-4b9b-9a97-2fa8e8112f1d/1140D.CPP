#include <bits/stdc++.h>
using namespace std;

/**
 * @file 1140D.CPP
 * @brief Solution for a geometry problem, "Minimum Triangulation".
 *
 * Problem Statement:
 * Given a regular n-sided polygon with vertices labeled 1 to `n` in order.
 * We need to triangulate this polygon by adding `n-3` non-intersecting
 * diagonals. The cost of a triangulation is the sum of the "weights" of the
 * `n-2` triangles formed. The weight of a triangle is the product of the labels
 * of its three vertices. The goal is to find the minimum possible cost of a
 * triangulation.
 *
 * Optimal Strategy (Greedy Approach):
 * To minimize a sum of products, we should try to keep the factors in each
 * product as small as possible. The vertex with the smallest label is '1'.
 * Therefore, an optimal strategy is to include vertex '1' in every triangle.
 *
 * This can be achieved by creating a "fan" triangulation, where we draw all
 * possible diagonals from vertex '1' to every other non-adjacent vertex. This
 * method is valid and forms the following `n-2` triangles:
 * (1, 2, 3)
 * (1, 3, 4)
 * (1, 4, 5)
 * ...
 * (1, n-1, n)
 *
 * The cost of this triangulation is the sum of the weights of these triangles:
 * Cost = (1 * 2 * 3) + (1 * 3 * 4) + ... + (1 * (n-1) * n)
 *
 * This can be written as the summation:
 * Cost = sum_{i=2}^{n-1} (1 * i * (i+1))
 *
 * The code implements this exact formula.
 *
 * Time Complexity: O(N) for the loop.
 * Space Complexity: O(1).
 */

// Common competitive programming macros for I/O optimization and brevity.
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"
#define forn(i, n) for (int i = 0; i < int(n); ++i)
#define vet(a)                                                                 \
  for (auto &i : a)                                                            \
  cout << i << " "
#define out(a) cout << a << endl

typedef long long int ll;

int32_t main() {
  IOS; // Apply I/O optimizations.

// Conditional compilation for local testing (reading/writing from files).
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n;
  cin >> n;

  ll ans = 0;
  // Block Logic: Calculate the total cost based on the optimal fan triangulation
  // strategy. The loop iterates from i=2 to n-1, summing up i * (i+1) for
  // each triangle in the fan.
  for (int i = 2; i < n; ++i) {
    // Each term i * (i+1) corresponds to the cost of triangle (1, i, i+1).
    // The multiplication by 1 is implicit.
    ans += i * 1ll * (i + 1);
  }
  cout << ans;

  return 0;
}
