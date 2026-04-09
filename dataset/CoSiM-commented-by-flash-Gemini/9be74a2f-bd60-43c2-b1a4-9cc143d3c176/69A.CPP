/**
 * @file 69A.CPP
 * @brief This program determines if a system of 'n' 3D force vectors is in equilibrium.
 * Equilibrium is achieved if the sum of all x-components, y-components, and z-components
 * of the forces individually equals zero. This is a fundamental concept in physics and
 * a common problem type in competitive programming.
 *
 * Algorithm: Vector sum and zero check (variant initialization).
 * 1. Read the number of vectors `n`.
 * 2. Read the components of the first vector (`r1`, `r2`, `r3`) and initialize accumulators with them.
 * 3. For the remaining `n-1` vectors:
 *    a. Read its x, y, and z components (`a`, `b`, `c`).
 *    b. Add `a` to `r1`, `b` to `r2`, and `c` to `r3`.
 * 4. After processing all vectors, check if `r1`, `r2`, and `r3` are all zero.
 * 5. Print "YES" if all sums are zero, "NO" otherwise.
 * Time Complexity: O(N) where N is the number of vectors, as each vector is processed in constant time.
 * Space Complexity: O(1) for storing accumulators and input variables.
 */

#include <iostream>

using namespace std;

/**
 * @brief Main function of the program.
 * Reads a series of 3D vectors and checks if their sum is the zero vector.
 * @pre Input starts with an integer `n`, followed by `n` lines, each containing three integers (x, y, z components).
 * @post Prints "YES" if the net force is zero, "NO" otherwise.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: 'n' for the number of vectors.
  // Functional Utility: 'a', 'b', 'c' for current vector components.
  // Functional Utility: 'r1', 'r2', 'r3' are accumulators for the total sum of each component.
  int n, a, b, c, r1, r2, r3;
  r1 = r2 = r3 = 0; // Functional Utility: Initializes accumulators to zero before reading the first vector.

  // Functional Utility: Reads the total number of force vectors.
  cin >> n;
  n--; // Functional Utility: Decrements 'n' because the first vector is read separately.

  // Functional Utility: Reads the components of the first vector and uses them to initialize the accumulators.
  cin >> r1 >> r2 >> r3;

  // Block Logic: Iterates through the remaining `n-1` force vectors.
  // Invariant: `r1`, `r2`, `r3` store the cumulative sum of components from vectors processed so far.
  while (n--) {
    // Functional Utility: Reads the x, y, and z components of the current vector.
    cin >> a >> b >> c;

    r1 += a; // Functional Utility: Accumulates the x-component.
    r2 += b; // Functional Utility: Accumulates the y-component.
    r3 += c; // Functional Utility: Accumulates the z-component.
  }

  // Block Logic: Checks if the total sum of all components is zero, indicating equilibrium.
  // Pre-condition: `r1`, `r2`, `r3` hold the final sums of all x, y, and z components respectively.
  if (!r1 && !r2 && !r3) // Functional Utility: Equivalent to (r1 == 0 && r2 == 0 && r3 == 0).
    cout << "YES" << endl; // Functional Utility: Prints "YES" if the system is in equilibrium.
  else
    cout << "NO" << endl; // Functional Utility: Prints "NO" if the system is not in equilibrium.

  return 0;
}
