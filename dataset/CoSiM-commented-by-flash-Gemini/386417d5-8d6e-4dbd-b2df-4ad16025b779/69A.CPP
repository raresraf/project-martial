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
  // Declare 'n' to store the number of force vectors.
  int n;
  // Read the total number of force vectors.
  cin >> n;

  // Declare and initialize variables for the sum of x, y, and z components to 0.
  int x = 0, y = 0, z = 0;

  // Block Logic: Read force components and accumulate sums.
  // Invariant: 'n' decreases with each iteration, processing one force vector per loop.
  while (n--) {
    // Declare temporary variables to store the x, y, and z components of the current force.
    int a, b, c;
    // Read the three components of the current force vector.
    cin >> a >> b >> c;

    // Accumulate the x-component into the total sum for x.
    x += a;
    // Accumulate the y-component into the total sum for y.
    y += b;
    // Accumulate the z-component into the total sum for z.
    z += c;
  }

  // Block Logic: Check for equilibrium using a ternary operator.
  // Pre-condition: 'x', 'y', 'z' hold the total sum of force components along each axis.
  (0 == x && x == y && y == z) // Condition: Are all three sums equal to zero?
      ? cout << "YES"          // If true, print "YES".
      : cout << "NO";          // If false, print "NO".
  cout << endl;                // Print a newline after the output.

  return 0; // Indicate successful program execution.
}
