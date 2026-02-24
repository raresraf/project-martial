/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - Young Physicist.
 *
 * This program determines if a system of 'n' forces acting on an object is in
 * equilibrium. A system of forces is in equilibrium if the vector sum of all
 * forces is zero. This means the sum of all x-components, y-components, and
 * z-components must each individually be zero.
 *
 * Algorithm: Summation of components. The program reads 'n' 3D force vectors,
 * sums their respective x, y, and z components, and then checks if all three
 * total sums are zero.
 * Time Complexity: O(N), where N is the number of forces.
 * Space Complexity: O(N) for storing the force components, though can be O(1)
 *                  if sums are calculated on the fly without storing all forces.
 */

#include <stdio.h> // For standard input/output functions like scanf and printf.

int main() {
  /**
   * @brief Main function to execute the Young Physicist problem logic.
   *
   * Reads an integer 'n' representing the number of force vectors.
   * Then, it reads 'n' sets of three integer values (x, y, z components)
   * for each force. It calculates the sum of x, y, and z components
   * separately and determines if the system is in equilibrium.
   */
  // Declare integer variables for loop counter 'i', number of forces 'n'.
  // Arrays 'x', 'y', 'z' store components of each force (max 105 forces).
  // 'sumx', 'sumy', 'sumz' accumulate the total force components.
  int i, n, x[105], y[105], z[105], sumx = 0, sumy = 0, sumz = 0;
  // Read the total number of forces.
  scanf("%d", &n);
  // Block Logic: Read force components.
  // Invariant: 'i' iterates from 0 to n-1, populating the x, y, z arrays.
  for (i = 0; i < n; i++)
    // Read the three components (x, y, z) for the i-th force.
    scanf("%d %d %d", &x[i], &y[i], &z[i]);
  // Block Logic: Sum up all force components.
  // Invariant: 'i' iterates from 0 to n-1, adding each force's components to their respective sums.
  for (i = 0; i < n; i++) {
    sumx += x[i]; // Add x-component to sumx.
    sumy += y[i]; // Add y-component to sumy.
    sumz += z[i]; // Add z-component to sumz.
  }
  // Block Logic: Check for equilibrium.
  // Pre-condition: 'sumx', 'sumy', 'sumz' contain the total force components.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    // If all sums are zero, the system is in equilibrium.
    printf("YES");
  else
    // Otherwise, the system is not in equilibrium.
    printf("NO");
  return 0; // Indicate successful program execution.
}