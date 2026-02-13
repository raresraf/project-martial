/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @author competitive-programmer
 *
 * @details This program checks if a system of 3D force vectors is in
 * equilibrium. A system is in equilibrium if the vector sum of all forces is
 * the zero vector, which means the sum of the components along each of the
 * three axes (x, y, z) must be zero.
 *
 * Algorithm (Optimal Single-Pass):
 * 1. Initialize three sum variables (`c1`, `c2`, `c3`) to 0.
 * 2. Read the number of vectors, `n`.
 * 3. Loop `n` times. In each iteration:
 *    a. Read the x, y, and z components of a vector.
 *    b. Immediately add each component to its respective sum variable.
 *    c. This avoids storing the entire set of vectors in memory.
 * 4. After the loop, check if all three sums are zero. If they are, the
 *    system is in equilibrium. Print "YES", otherwise print "NO".
 *
 * Time Complexity: O(n), where n is the number of vectors.
 * Space Complexity: O(1).
 */
#include <iostream>

int main() {
  int n;

  // c1, c2, c3 will store the sum of the x, y, and z components respectively.
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;

  // A temporary variable to read each component.
  int temp;

  std::cin >> n;

  /**
   * Block Logic: This loop processes each of the 'n' vectors in a single pass,
   * accumulating the sums of the components without storing the vectors.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> temp;
    c1 += temp;

    std::cin >> temp;
    c2 += temp;

    std::cin >> temp;
    c3 += temp;
  }

  /**
   * Decision Logic: Use a ternary operator to check the equilibrium condition.
   * `!c1` is true if c1 is 0. The condition checks if all three sums are zero.
   */
  std::cout << ((!c1 && !c2 && !c3) ? "YES" : "NO") << std::endl;

  return 0;
}
