/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (69A).
 * @details The problem determines if a body is in equilibrium under the influence
 * of multiple force vectors. In physics, a body is in equilibrium if the vector
 * sum of all forces acting upon it is zero.
 *
 * @algorithm
 * The solution calculates the net force by summing the components of all individual
 * force vectors.
 * 1. It initializes three variables (`sx`, `sy`, `sz`) to store the sum of the
 *    x, y, and z components of all vectors, respectively.
 * 2. It reads the number of force vectors, `n`.
 * 3. It loops `n` times, reading each 3D force vector `(x, y, z)`.
 * 4. In each iteration, it adds the components of the current vector to the
 *    running sums (`sx`, `sy`, `sz`).
 * 5. After the loop, it checks if the resultant vector is the zero vector, i.e.,
 *    if `sx`, `sy`, and `sz` are all zero.
 * 6. If they are all zero, the body is in equilibrium, and "YES" is printed.
 *    Otherwise, "NO" is printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of force vectors, due to the
 * single loop that processes each vector.
 * Space Complexity: O(1), as it uses a fixed number of variables to store the sums.
 */
#include <iostream>

using namespace std;

int main() {
  /**
   * @variable n: The number of force vectors.
   * @variable x, y, z: Components of a single force vector.
   * @variable sx, sy, sz: The components of the resultant (sum) force vector.
   */
  int n;
  int x, y, z;
  int sx = 0, sy = 0, sz = 0;
  
  // Read the number of vectors.
  scanf("%d", &n);
  
  /**
   * @block
   * @brief This loop reads each force vector and adds its components to the total sum.
   * @invariant `sx`, `sy`, `sz` hold the vector sum of all forces processed so far.
   */
  while (n--) {
    scanf("%d %d %d", &x, &y, &z);
    sx += x;
    sy += y;
    sz += z;
  }
  
  /**
   * @block
   * @brief Checks if the net force is zero, which is the condition for equilibrium.
   */
  if (sx == 0 && sy == 0 && sz == 0) {
    printf("YES");
  } else {
    printf("NO");
  }
}