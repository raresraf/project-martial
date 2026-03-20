/**
 * @file 69A.CPP
 * @brief Solution for Codeforces problem 69A, "Young Physicist".
 * @details The program determines if a body is in equilibrium given a set of
 * `n` force vectors in 3D space.
 *
 * @section Algorithm
 * The principle of equilibrium in physics states that a body is in equilibrium if
 * the vector sum of all forces acting upon it is zero. In a 3D Cartesian system,
 * this means the sum of the components of all force vectors along the x, y, and z
 * axes must each be zero.
 *
 * The algorithm implements this principle directly:
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize three variables (`is`, `js`, `ks`) to zero, to accumulate the
 *    sum of forces along the x, y, and z axes, respectively.
 * 3. Loop `n` times, reading the three components of each force vector.
 * 4. Add each component to its corresponding accumulator.
 * 5. After the loop, check if all three accumulators are zero.
 * 6. If `is == 0`, `js == 0`, and `ks == 0`, the net force is zero, and the
 *    program prints "YES". Otherwise, it prints "NO".
 *
 * @subsection Complexity
 * - Time Complexity: O(n), where `n` is the number of vectors, due to the single
 *   loop that processes each vector.
 * - Space Complexity: O(1), as the memory usage is constant regardless of `n`.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  // is, js, ks: accumulators for the sum of x, y, and z components of the forces.
  int is = 0, js = 0, ks = 0;
  // l, p, q: variables to store the x, y, and z components of a single force vector.
  int l, p, q;

  /**
   * @brief This loop iterates through `n` force vectors, summing their components.
   * Pre-condition: The component sums `is`, `js`, `ks` are initialized to 0.
   * Invariant: At the end of each iteration, `is`, `js`, and `ks` hold the
   * component-wise sum of all vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> l >> p >> q;
    is += l;
    js += p;
    ks += q;
  }

  /**
   * @brief Checks for equilibrium by testing if the net force vector is (0,0,0).
   * In C++, a non-zero integer evaluates to true, and zero evaluates to false.
   * Therefore, `!is` is a concise way to check if `is == 0`.
   */
  if (!is && !js && !ks) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}