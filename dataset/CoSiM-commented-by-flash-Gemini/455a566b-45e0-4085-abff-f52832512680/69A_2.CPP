/**
 * @file 69A_2.CPP
 * @brief Solution to a competitive programming problem (Codeforces 69A - "Young Physicist").
 *
 * This program determines if a body is in equilibrium given `n` forces acting upon it.
 * A body is in equilibrium if the sum of all vectors acting on it is the zero vector.
 * This translates to checking if the sum of all x-components, the sum of all y-components,
 * and the sum of all z-components are all simultaneously zero.
 *
 * Algorithm: Vector summation and zero check.
 * It reads `n` three-dimensional vectors (forces). It maintains three separate sums
 * for the x, y, and z components. After reading all vectors, it checks if all three
 * accumulated sums are equal to zero.
 *
 * Time Complexity: O(N), where N is the number of forces/vectors.
 * The program iterates through the `n` forces exactly once.
 * Space Complexity: O(1), as it only uses a fixed number of integer variables for sums and components.
 */
#include <stdio.h> // Required for C-style input/output operations (scanf, printf).

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads `n` 3D force vectors, sums their components, and determines if the body is in equilibrium.
 */
int main() {
  int n;                                   // Variable to store the number of forces.
  int sum1 = 0, sum2 = 0, sum3 = 0;        // Accumulators for x, y, and z components respectively.
  int a, b, c;                             // Variables to store the components of each force vector.

  // Pre-condition: Read the number of force vectors 'n'.
  scanf("%d", &n);
  // Block Logic: Iterate 'n' times, once for each force vector.
  // Invariant: 'n' decreases with each iteration, loop continues until all vectors are processed.
  while (n--) {
    // Pre-condition: Read the three components (x, y, z) of the current force vector.
    scanf("%d %d %d", &a, &b, &c);
    sum1 += a; // Add x-component to sum1.
    sum2 += b; // Add y-component to sum2.
    sum3 += c; // Add z-component to sum3.
  }
  // Block Logic: Check if the sum of all components in each dimension is zero.
  // Functional Utility: This condition determines if the body is in equilibrium.
  if (sum1 == 0 && sum2 == 0 && sum3 == 0) {
    printf("YES\n"); // If all sums are zero, the body is in equilibrium.
  } else
    printf("NO\n");  // Otherwise, the body is not in equilibrium.
  return 0;          // Indicate successful execution.
}
