/**
 * @file 69A.CPP
 * @brief Solution for competitive programming problem: "Young Physicist".
 *
 * @details This program determines if a body is in equilibrium given a set of 'n'
 *          forces acting on it. A body is in equilibrium if the sum of the
 *          x-components, y-components, and z-components of all forces is zero.
 *          Algorithm: Summation of corresponding vector components.
 *          Time Complexity: O(N), where N is the number of forces, as each force's
 *                           components are read and added once.
 *          Domain: Competitive Programming.
 */
#include "bits/stdc++.h"
// Macro to disable synchronization with C stdio and untie cin/cout from cout/cin for faster I/O.
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
// Macro for file I/O redirection (commented out for standard console I/O).
#define FIO                                                                    \
  freopen("input.txt", "r", stdin);                                            \
  freopen("output.txt", "w", stdout);

using namespace std;

int main(void) {
  do_not_sync // Apply fast input/output optimization.
      // FIO // Commented out: file input/output redirection.
      
  int n; // Declare 'n' for the number of forces.
  cin >> n; // Read the number of forces.
  // Declare accumulators for the sum of x, y, and z components.
  int sumx = 0, sumy = 0, sumz = 0;

  // Block Logic: Iterate 'n' times, reading each force's components and accumulating their sums.
  // Invariant: 'sumx', 'sumy', 'sumz' store the running sum of x, y, z components respectively.
  for (int i = 0; i < n; i++) {
    int x, y, z; // Declare temporary variables for current force components.
    cin >> x >> y >> z; // Read the x, y, and z components of the current force.
    sumx += x; // Accumulate x-component sum.
    sumy += y; // Accumulate y-component sum.
    sumz += z; // Accumulate z-component sum.
  }

  // Block Logic: Determine if the body is in equilibrium.
  // Precondition: A body is in equilibrium if the sum of all components in each dimension is zero.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl; // Output "YES" if in equilibrium.
  else
    cout << "NO" << endl;  // Output "NO" otherwise.

  return 0; // Indicate successful program termination.
}
