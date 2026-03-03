/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 * This program determines if a body is in equilibrium given a set of N forces acting upon it.
 * A body is in equilibrium if the vector sum of all forces acting on it is zero.
 * This translates to the sum of all X-components, Y-components, and Z-components being zero.
 * Algorithm: Read all force components into arrays, then sum them up and check if all sums are zero.
 * Time Complexity: O(N) - where N is the number of forces. The program iterates through the forces twice.
 * Space Complexity: O(N) - Three arrays of size N are used to store the force components.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries
using namespace std;     // Uses the standard namespace

int main() {
  // Commented out: `freopen` is typically used for redirecting standard input/output to files
  // for local testing, but it's usually removed or conditionally compiled for online submission.
  // freopen("file.text", "r", stdin);
  
  int n; // Declares an integer to store the number of forces.
  cin >> n; // Reads the number of forces from standard input.
  
  // Declares three arrays to store the x, y, and z components of each force.
  // The size `n + 1` is used to avoid potential off-by-one errors with 1-based indexing,
  // although the code uses 0-based indexing for input.
  int x[n + 1], y[n + 1], z[n + 1];

  /**
   * Block Logic: Reads the x, y, and z components for each of the 'n' forces.
   * Invariant: After this loop, arrays `x`, `y`, `z` contain all force components.
   */
  for (int i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i]; // Reads components for the i-th force.

  // Initializes accumulators for the sum of x, y, and z components of all forces.
  int sumX = 0, sumY = 0, sumZ = 0;
  
  /**
   * Block Logic: Sums up the x, y, and z components from the respective arrays.
   * Invariant: After this loop, `sumX`, `sumY`, and `sumZ` hold the total sum
   *            of all x, y, and z components, respectively.
   */
  for (int i = 0; i < n; i++) {
    sumX += x[i]; // Adds the x-component of the i-th force to sumX.
    sumY += y[i]; // Adds the y-component of the i-th force to sumY.
    sumZ += z[i]; // Adds the z-component of the i-th force to sumZ.
  }

  // Decision Logic: Checks if the body is in equilibrium.
  // The body is in equilibrium if and only if the sum of all x, y, and z components are all zero.
  if (sumX == 0 && sumY == 0 && sumZ == 0)
    cout << "YES" << endl; // If all sums are zero, the body is in equilibrium.
  else
    cout << "NO" << endl;  // Otherwise, the body is not in equilibrium.

  return 0; // Indicates successful program execution.
}
