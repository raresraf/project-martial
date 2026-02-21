/**
 * @file 69A.CPP
 * @brief Checks for equilibrium in a system of 3D force vectors.
 *
 * This program reads a number of 3D vectors that represent forces and
 * determines if the system is in static equilibrium. Equilibrium is achieved if
 * the vector sum of all forces is the zero vector, meaning the sum of components
 * along each of the three axes (x, y, z) is zero.
 *
 * Algorithm:
 * 1. Read an integer `n`, the number of force vectors.
 * 2. Initialize three integer variables, `x`, `y`, and `z`, to zero. These will
 *    store the running sum of the vector components along each axis.
 * 3. Loop `n` times to process each vector.
 *    a. In each iteration, read the three components (`a`, `b`, `c`) of a vector.
 *    b. Add these components to the respective sum variables (`x`, `y`, `z`).
 * 4. After the loop, check if `x`, `y`, and `z` are all equal to zero.
 * 5. If they are, print "YES" to indicate equilibrium.
 * 6. Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the number of vectors, as the program
 * iterates through the input vectors only once.
 * Space Complexity: O(1), as it uses a constant amount of memory regardless of
 * the number of vectors, by not storing the individual vector data.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a, b, c, x = 0, y = 0, z = 0;
  // Read the quantity of vectors.
  cin >> n;
  
  /**
   * @brief Iterates through each vector, accumulating the component sums.
   * Pre-condition: `n` is the number of vectors to process.
   * Invariant: `x`, `y`, `z` hold the sum of components for vectors processed so far.
   */
  for (int i = 0; i < n; ++i) {
    // Read the three components of the current vector.
    cin >> a >> b >> c;
    // Add components to the total sums.
    x += a;
    y += b;
    z += c;
  }
  
  /**
   * @brief Determine if the system is in equilibrium.
   * The condition for equilibrium is that the net force along each axis is zero.
   */
  if ((x == 0) && (y == 0) && (z == 0))
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}