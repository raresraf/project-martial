/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem involving vector sum.
 *
 * Algorithm: Iterative summation of vector components.
 * Time Complexity: O(N), where N is the number of vectors.
 * Space Complexity: O(1).
 *
 * This program determines if a system of N 3D vectors (forces) is in equilibrium.
 * It reads N, then N sets of (x, y, z) components, summing each component type.
 * If all resultant sums are zero, the system is in equilibrium.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  // Functional Utility: Reads the number of vectors (forces) from standard input.
  std::cin >> n;
  int x, y, z;
  int t1 = 0, t2 = 0, t3 = 0;

  /**
   * Block Logic: Iterates N times to read and accumulate the components of each vector.
   * Pre-condition: 'n' contains the total number of vectors to process.
   * Invariant: At each iteration, 't1', 't2', 't3' store the cumulative sums of x, y, and z components
   *            for all vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the x, y, and z components of the current vector.
    std::cin >> x >> y >> z;
    // Functional Utility: Accumulates the x-component.
    t1 += x;
    // Functional Utility: Accumulates the y-component.
    t2 += y;
    // Functional Utility: Accumulates the z-component.
    t3 += z;
  }

  /**
   * Block Logic: Checks if the system of vectors is in equilibrium.
   * Pre-condition: 't1', 't2', 't3' contain the total sum of x, y, and z components, respectively.
   * Invariant: The decision is based solely on whether all three resultant sums are zero.
   */
  if (t1 == 0 && t2 == 0 && t3 == 0) {
    // Functional Utility: Indicates that the system is in equilibrium.
    std::cout << "YES" << std::endl;
  } else {
    // Functional Utility: Indicates that the system is not in equilibrium.
    std::cout << "NO" << std::endl;
  }
  return 0;
}
