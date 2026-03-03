/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 * This program determines if a body is in equilibrium given a set of N forces acting upon it.
 * A body is in equilibrium if the vector sum of all forces acting on it is zero.
 * This translates to the sum of all X-components, Y-components, and Z-components being zero.
 * Algorithm: Reads all force components into a 2D array, then sums components along each axis.
 * Time Complexity: O(N) - where N is the number of forces. The program iterates through the forces
 *                  to read input and again to sum up components.
 * Space Complexity: O(N) - A 2D array of size N x 3 is used to store the force components.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

int main() {
  // n: total number of forces.
  // i, j: loop counters.
  // s: accumulates the sum of components for the current axis.
  // a[100][3]: a 2D array to store the x, y, z components of up to 100 forces.
  // f: a flag (1 for YES, 0 for NO), initialized to 1 (assuming equilibrium initially).
  int n, i, j, s = 0, a[100][3], f = 1;
  
  cin >> n; // Reads the number of forces from standard input.
  
  /**
   * Block Logic: Reads the x, y, and z components for each of the 'n' forces.
   * Invariant: After this loop, the `a` array contains all force components.
   */
  for (i = 0; i < n; i++) // Loop for each force.
    for (j = 0; j < 3; j++) // Loop for each component (x, y, z).
      cin >> a[i][j]; // Reads a component and stores it in the 2D array.
  
  /**
   * Block Logic: Calculates the sum of components along each axis (x, y, and z).
   * If any sum is non-zero, the body is not in equilibrium, and the flag 'f' is set to 0.
   * Invariant: `f` remains 1 only if all component sums are zero.
   */
  for (j = 0; j < 3; j++) { // Outer loop for each component axis (j=0 for X, j=1 for Y, j=2 for Z).
    s = 0;                   // Reset sum for the current axis.
    for (i = 0; i < n; i++)  // Inner loop to sum components for the current axis across all forces.
      s += a[i][j];          // Adds the component of the i-th force for the j-th axis to 's'.
    if (s) { // Checks if the sum for the current axis is non-zero.
      f = 0; // If non-zero, set flag to 0 (not in equilibrium).
      break; // No need to check other axes, break from the outer loop.
    }
  }
  
  // Decision Logic: Prints "YES" if the body is in equilibrium, "NO" otherwise.
  if (f) // If the flag 'f' is still 1 (meaning all component sums were zero).
    cout << "YES";
  else
    cout << "NO";

  return 0; // Indicates successful program execution.
}
