/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" competitive programming problem (Codeforces 69A).
 * The problem states that an object is in equilibrium if the sum of all forces acting on it is zero.
 * The program reads the number of forces and then the x, y, and z components for each force.
 * It determines if the object is in equilibrium by checking if the sum of x-components,
 * sum of y-components, and sum of z-components are all independently equal to zero.
 * Algorithm: Iterative summation of force components with conditional equilibrium check.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

int main() {
  // Declares 'n' for the number of force vectors.
  // Declares 'y', 'x', 'z' for the components of each force vector.
  // Declares 'suma', 'sumb', 'sumc' to accumulate the sums of x, y, and z components respectively.
  int n, y, x, z, suma = 0, sumb = 0, sumc = 0;
  // Reads the number of force vectors.
  cin >> n;
  
  /**
   * Block Logic: Iterates 'n' times (or until 'n' becomes 0), once for each force vector.
   * Invariant: At the start of each iteration, 'n' represents the remaining force vectors to process,
   * and 'suma', 'sumb', 'sumc' hold the accumulated sums of components so far.
   * Pre-condition: 'n' is a positive integer.
   * Post-condition: 'suma', 'sumb', 'sumc' hold the total sums of x, y, and z components.
   */
  while (n--) {
    // Reads the x, y, and z components of the current force vector.
    cin >> x >> y >> z;
    // Accumulates the sums for each component.
    suma = suma + x;
    sumb = sumb + y;
    sumc = sumc + z;
  }
  
  /**
   * Conditional Logic: Checks if the object is in equilibrium.
   * An object is in equilibrium if and only if the sum of all force components
   * in each dimension (x, y, z) is zero.
   * The condition `suma || sumb || sumc` evaluates to true if any of the sums are non-zero.
   * So, if any sum is non-zero, it means not in equilibrium.
   */
  if (suma || sumb || sumc)
    cout << "NO" << endl; // Prints "NO" if not in equilibrium.
  else
    cout << "YES" << endl; // Prints "YES" if in equilibrium.
  
  return 0; // Indicates successful program execution.
}
