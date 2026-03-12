/**
 * @6116f459-b034-424c-bd20-2cf8243226c0/69A.CPP
 * @brief This program solves problem 69A. It takes 'n' three-dimensional vectors as input.
 * The objective is to determine if the sum of all x-components, all y-components, and
 * all z-components are simultaneously zero. This effectively checks if a system of forces
 * is in equilibrium.
 *
 * Algorithm: Two-pass linear scan to read components and then sum them, followed by a conditional check.
 * Time Complexity: O(N) where N is the number of vectors.
 * Space Complexity: O(N) for storing the components of 'n' vectors.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, vector.

using namespace std; // Uses the standard namespace.

int main() {
  int n; // Declares 'n', representing the number of 3D vectors.
  cin >> n; // Reads 'n' from standard input.
  
  // Declares three arrays of size 'n' to store the x, y, and z components of each vector.
  // Initializes sumx, sumy, sumz to 0, which will accumulate the sums of components.
  int x[n], y[n], z[n], sumx = 0, sumy = 0, sumz = 0; 

  /**
   * Block Logic: First pass - reads 'n' sets of (x, y, z) components.
   * Invariant: After this loop, arrays 'x', 'y', 'z' are populated with vector components.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each vector.
    cin >> x[i] >> y[i] >> z[i]; // Reads the three components of the current vector.
  }

  /**
   * Block Logic: Second pass - calculates the sum of all x-components, y-components,
   * and z-components separately.
   * Invariant: After this loop, 'sumx', 'sumy', 'sumz' hold the resultant sums of each dimension.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each vector.
    sumx = sumx + x[i]; // Inline: Adds the x-component to the total sum of x-components.
    sumy = sumy + y[i]; // Inline: Adds the y-component to the total sum of y-components.
    sumz = sumz + z[i]; // Inline: Adds the z-component to the total sum of z-components.
  }

  /**
   * Block Logic: Checks if the system of forces is in equilibrium.
   * Pre-condition: 'sumx', 'sumy', 'sumz' contain the final resultant sums.
   * Invariant: Prints "YES" if all sums are zero, otherwise "NO".
   */
  if (sumx == 0 && sumy == 0 && sumz == 0) // If all three resultant sums are zero.
    cout << "YES" << endl; // Prints "YES".
  else
    cout << "NO" << endl; // Prints "NO".
  
  // The program implicitly returns 0 on successful completion of main.
}
