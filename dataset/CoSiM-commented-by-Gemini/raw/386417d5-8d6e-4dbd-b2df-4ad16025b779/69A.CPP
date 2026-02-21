/**
 * @file 69A.CPP
 * @brief Determines if a system of 3D force vectors is in equilibrium.
 *
 * This program reads a set of 3D vectors representing forces. It calculates
 * the net force by summing up all the vectors. If the resultant vector is
 * the zero vector (i.e., the sums of the x, y, and z components are all zero),
 * the system is in equilibrium, and the program outputs "YES". Otherwise, it
 * outputs "NO".
 *
 * Algorithm:
 * 1. Read the total number of vectors, `n`.
 * 2. Initialize three variables `x`, `y`, `z` to 0 to store the sum of components.
 * 3. Loop `n` times, decrementing `n` in each iteration.
 *    a. In each loop, read the three components of a vector (`a`, `b`, `c`).
 *    b. Add these components to the running totals `x`, `y`, and `z`.
 * 4. After the loop, check if all three sums (`x`, `y`, `z`) are equal to zero.
 * 5. Print "YES" for equilibrium, "NO" otherwise.
 *
 * Time Complexity: O(N), where N is the number of vectors, because the program
 * iterates through the vectors once.
 * Space Complexity: O(1), as the vectors are not stored; only their running sum is maintained.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  // Read the number of vectors.
  cin >> n;

  // Initialize variables to hold the sum of components of all vectors.
  int x = 0, y = 0, z = 0;

  /**
   * @brief Loop `n` times to read vectors and compute the net force.
   * Pre-condition: `n` contains the number of vectors.
   * Invariant: After each iteration, `x`, `y`, and `z` contain the component-wise
   * sum of the vectors read so far.
   */
  while (n--) {
    int a, b, c;
    // Read the components of one vector.
    cin >> a >> b >> c;

    // Add the components to the running sum.
    x += a;
    y += b;
    z += c;
  }

  /**
   * @brief Check for equilibrium and print the result.
   * The condition `0 == x && x == y && y == z` is a concise way to check if
   * x, y, and z are all zero. It first checks `0 == x`, then `x == y` (which is
   * `0 == y`), and finally `y == z` (which is `0 == z`).
   */
  (0 == x && x == y && y == z) ? cout << "YES" : cout << "NO";
  cout << endl;

  return 0;
}
