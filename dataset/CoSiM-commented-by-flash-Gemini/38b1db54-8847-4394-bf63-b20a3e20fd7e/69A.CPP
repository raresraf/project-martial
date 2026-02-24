/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - Young Physicist.
 *
 * This program determines if a system of 'n' forces acting on an object is in
 * equilibrium. A system of forces is in equilibrium if the vector sum of all
 * forces is zero. This means the sum of all x-components, y-components, and
 * z-components must each individually be zero.
 *
 * Algorithm: On-the-fly summation of components. The program reads 'n' 3D force
 * vectors and directly accumulates their x, y, and z components into running sums.
 * Finally, it checks if all three total sums are zero.
 * Time Complexity: O(N), where N is the number of forces.
 * Space Complexity: O(1), as force components are summed on the fly without
 *                  storing all individual force vectors.
 */

#include <bits/stdc++.h> // Includes various standard libraries (e.g., iostream, vector, algorithm).
using namespace std;     // Uses the standard namespace to avoid prefixing std::.

int main() {
  /**
   * @brief Main function to execute the Young Physicist problem logic.
   *
   * Reads an integer 'n' representing the number of force vectors.
   * It then iterates 'n' times, reading the three components (x, y, z) of
   * each force and accumulating them into total sums for each axis.
   * Finally, it checks if all three total sums are zero to determine
   * equilibrium and prints "YES" or "NO".
   */
  // Declare 'n' for the number of force vectors.
  // Declare 'a', 'b', 'c' for the components of each force vector.
  // Initialize 'x', 'y', 'z' to 0 for accumulating the sums of components.
  int n, a, b, c, x = 0, y = 0, z = 0;
  // Read the total number of force vectors.
  cin >> n;
  // Block Logic: Read force components and accumulate sums.
  // Invariant: 'i' iterates from 0 to n-1, processing one force vector in each iteration.
  for (int i = 0; i < n; ++i) {
    // Read the three components (a, b, c) of the current force vector.
    cin >> a >> b >> c;
    // Accumulate the components into their respective sums.
    x += a;
    y += b;
    z += c;
  }
  // Block Logic: Check for equilibrium.
  // Pre-condition: 'x', 'y', 'z' hold the total sum of force components along each axis.
  if ((x == 0) && (y == 0) && (z == 0))
    // If all three sums are zero, the system is in equilibrium.
    cout << "YES\n";
  else
    // Otherwise, the system is not in equilibrium.
    cout << "NO\n";

  return 0; // Indicate successful program execution.
}