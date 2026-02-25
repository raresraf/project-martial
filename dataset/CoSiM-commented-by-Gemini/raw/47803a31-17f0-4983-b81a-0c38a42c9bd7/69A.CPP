/**
 * @file 69A.CPP
 * @brief Solution for a physics problem, "Young Physicist," from a programming contest.
 * @details The problem is to determine if a body is in equilibrium under the influence
 * of several forces. In physics, a body is in equilibrium if and only if the vector
 * sum of all forces acting upon it is zero.
 *
 * Algorithm:
 * 1. The vector sum of forces is zero if, and only if, the sum of the components
 *    of the forces along each coordinate axis (x, y, z) is zero.
 * 2. Initialize sums for each axis (sum_x, sum_y, sum_z) to 0.
 * 3. Read the number of forces, 'n'.
 * 4. Loop 'n' times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add these components to their respective axis sums.
 * 5. After processing all forces, check if sum_x, sum_y, and sum_z are all zero.
 * 6. If they are, the body is in equilibrium ("YES"). Otherwise, it is not ("NO").
 *
 * Time Complexity: O(n), where 'n' is the number of force vectors.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int sumx, sumy, sumz, xi, yi, zi, n;
  sumx = 0;
  sumy = 0;
  sumz = 0;
  cin >> n;

  /**
   * @block
   * @description This loop reads 'n' force vectors and accumulates their
   * components into three separate sums, one for each axis.
   * Invariant: After each iteration, sumx, sumy, and sumz hold the net
   * force components for the vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> xi >> yi >> zi;
    sumx = sumx + xi;
    sumy = sumy + yi;
    sumz = sumz + zi;
  }
  
  /**
   * @block
   * @description Checks for the equilibrium condition. The body is in equilibrium
   * if the net force along each of the three axes is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}