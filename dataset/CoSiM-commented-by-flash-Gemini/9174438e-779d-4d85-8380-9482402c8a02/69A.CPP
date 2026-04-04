/**
 * @file 69A.CPP
 * @brief Determines if a system of forces is in equilibrium.
 *
 * This program reads an integer 'n' representing the number of forces acting on an object.
 * For each force, it reads three integer components (Fx, Fy, Fz). It then calculates
 * the net force along each axis (sum of Fx, sum of Fy, sum of Fz). If all three net
 * force components are zero, the system is in equilibrium, and the program outputs "YES".
 * Otherwise, it outputs "NO".
 *
 * Algorithm: Vector summation. Reads 'n' 3D vectors and sums their components.
 * Time Complexity: O(N), where N is the number of forces, as each force is processed once.
 * Space Complexity: O(1).
 */
// Sagindyk Urazayev
// http://codeforces.com/problemset/problem/69/A

#include <iostream>

using namespace std;

int main() {
  /**
   * @brief Main function: Determines if a system of 'n' forces is in equilibrium.
   * @return 0 if the program executes successfully.
   */
  int zres, yres, xres, n, i, x, y, z; // Declare variables for sum of components (xres, yres, zres),
                                     // number of forces (n), loop counter (i), and individual force components (x, y, z).
  xres = yres = zres = 0; // Initialize sum of components to zero.
  cin >> n;             // Read the number of forces 'n'.
  /**
   * @brief Loop to read 'n' sets of force components and accumulate their sums.
   * @pre 'n' is a positive integer. 'xres', 'yres', 'zres' are initialized to 0.
   * @post 'xres', 'yres', 'zres' contain the total sum of force components along x, y, and z axes respectively.
   * @invariant 'i' iterates from 0 to n-1, processing one force vector in each iteration.
   */
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z; // Read the x, y, and z components of the current force.
    xres = xres + x;     // Add x-component to total x-sum.
    yres = yres + y;     // Add y-component to total y-sum.
    zres = zres + z;     // Add z-component to total z-sum.
  }
  /**
   * @brief Conditional block to check if the net force in all directions is zero.
   * @pre 'xres', 'yres', 'zres' hold the accumulated sums of force components.
   * @invariant If all three sums are zero, the object is in equilibrium.
   */
  if ((xres == 0) && (yres == 0) && (zres == 0)) { // Check for equilibrium condition.
    cout << "YES" << endl; // Output "YES" if the system is in equilibrium.
  } else {
    cout << "NO" << endl; // Output "NO" otherwise.
  }
  return 0; // Return 0 for successful execution.
}
