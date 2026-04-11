/**
 * @file 69A.CPP
 * @brief This program determines if a system of N 3-dimensional vectors is in equilibrium.
 * It reads N vectors, each with x, y, and z components, and checks if the
 * sum of all x-components, all y-components, and all z-components are individually zero.
 *
 * Algorithm: Vector sum calculation. Iteratively accumulate sums of components, then check for zero totals.
 * Time Complexity: O(N) where N is the number of vectors.
 * Space Complexity: O(1) as only a fixed number of variables store the sums.
 */

#include <iostream>
using namespace std;

int main() {
  // Functional Utility: 'n' stores the total number of vectors to process.
  int n;
  // Block Logic: Reads the number of vectors 'n' from standard input.
  cin >> n;

  // Functional Utility: 'x', 'y', 'z' temporarily store the components of each vector.
  int x, y, z;
  // Functional Utility: 'tx', 'ty', 'tz' accumulate the sum of x, y, and z components respectively.
  //                     Initialized to 0.
  int tx = 0, ty = 0, tz = 0;

  /**
   * Block Logic: Iterates through each vector to read its components and accumulate sums.
   * Invariant: At the start of each iteration, 'n' represents the remaining vectors to process.
   */
  while (n--) {
    // Block Logic: Reads the x, y, and z components of the current vector.
    cin >> x >> y >> z;
    // Functional Utility: Accumulates the x, y, and z components into their respective total sums.
    tx += x, ty += y, tz += z;
  }

  /**
   * Block Logic: Checks if the system of vectors is in equilibrium.
   * Pre-condition: 'tx', 'ty', 'tz' hold the total sums of all components.
   * Invariant: If all three sums are exactly zero, the system is in equilibrium ("YES"), otherwise ("NO").
   */
  if (tx == 0 && ty == 0 && tz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
