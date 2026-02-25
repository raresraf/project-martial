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

#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace to avoid prefixing std::.

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
  // Declare and initialize variables for the sum of x, y, and z components to 0.
  // 'n' stores the number of force vectors.
  int sumx = 0, sumy = 0, sumz = 0, n;
  // Read the total number of force vectors.
  cin >> n;
  // Block Logic: Read force components and accumulate sums.
  // Invariant: 'i' iterates 'n' times, processing one force vector in each iteration.
  for (int i = 0; i < n; i++) {
    // Declare temporary variables to store the x, y, and z components of the current force.
    int x, y, z;
    // Read the three components of the current force vector.
    cin >> x >> y >> z;
    // Accumulate the x-component into the total sum for x.
    sumx += x;
    // Accumulate the y-component into the total sum for y.
    sumy += y;
    // Accumulate the z-component into the total sum for z.
    sumz += z;
  }
  // Block Logic: Check for equilibrium.
  // Pre-condition: 'sumx', 'sumy', 'sumz' hold the total sum of force components along each axis.
  if ((sumx == 0) && (sumy == 0) && (sumz == 0)) {
    // If all three sums are zero, the system is in equilibrium.
    cout << "YES" << endl;
  } else
    // Otherwise, the system is not in equilibrium.
    cout << "NO" << endl;
  return 0; // Indicate successful program execution.
}