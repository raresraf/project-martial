/**
 * @file 69A.CPP
 * @brief Checks for equilibrium in a system of 3D forces (vectors).
 *
 * This program reads `n` vectors from standard input, representing forces. It then
 * calculates the net force by summing all the individual force vectors. If the
 * resultant net force vector is the zero vector (i.e., its x, y, and z components
 * are all zero), the system is considered to be in equilibrium.
 *
 * @algorithm
 * 1. Read the number of vectors, `n`.
 * 2. Declare three arrays, `x`, `y`, and `z`, to store the components of the `n` vectors.
 *    Note: This approach uses Variable Length Arrays (VLAs), which are a non-standard C++ extension.
 * 3. Loop from `i = 0` to `n-1` to read all vector components and populate the arrays.
 * 4. Initialize three sum variables, `sumX`, `sumY`, `sumZ`, to zero.
 * 5. Loop from `i = 0` to `n-1` again to calculate the sum of each component.
 *    a. `sumX` accumulates the sum of all x-components.
 *    b. `sumY` accumulates the sum of all y-components.
 *    c. `sumZ` accumulates the sum of all z-components.
 * 6. After summing, check if `sumX`, `sumY`, and `sumZ` are all equal to 0.
 * 7. If they are all zero, print "YES" to indicate equilibrium.
 * 8. Otherwise, print "NO".
 *
 * @implementation_note
 * This implementation first stores all vector components in arrays and then iterates
 * through them a second time to compute the sum. A more memory-efficient approach
 * would be to sum the components as they are read, without storing them in arrays,
 * reducing the space complexity from O(n) to O(1).
 *
 * @complexity
 * Time Complexity: O(n), due to two separate loops that each run `n` times.
 * Space Complexity: O(n), because it uses three arrays of size `n` to store the vector components.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // The commented-out line is often used in competitive programming to redirect input from a file for testing.
  // freopen("file.text", "r", stdin);
  
  // `n` is the total number of vectors.
  int n;
  cin >> n;
  
  // Using Variable Length Arrays (VLAs) to store the components of each vector.
  // This is a non-standard C++ feature, but supported by some compilers like GCC.
  int x[n], y[n], z[n];

  // First loop: Read and store all vector components.
  for (int i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];

  // Initialize variables to store the sum of each vector component.
  int sumX = 0, sumY = 0, sumZ = 0;
  
  /**
   * @brief Second loop: Calculate the sum of components for the resultant vector.
   *
   * Invariant: At the start of each iteration, sumX, sumY, and sumZ hold the component
   * sums of the first `i` vectors.
   */
  for (int i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumZ += z[i];
  }

  // Check for the equilibrium condition: the net force vector must be the zero vector.
  if (sumX == 0 && sumY == 0 && sumZ == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
