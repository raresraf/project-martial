/**
 * @file 69A.CPP
 * @brief Checks if a body is in equilibrium given a set of force vectors.
 *
 * This program solves the "Young Physicist" problem. It determines if a body
 * is in equilibrium by checking if the vector sum of all applied forces is zero.
 * The input consists of N 3D force vectors.
 *
 * Algorithm:
 * 1. Read the number of force vectors, n.
 * 2. Initialize sum variables for x, y, and z components to zero.
 * 3. Loop n times, reading each force vector (x, y, z).
 * 4. Add the components of the read vector to their respective sums.
 * 5. After the loop, if the sums of all three components are zero, the body is
 *    in equilibrium (output "YES"). Otherwise, it is not (output "NO").
 *
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1), as it uses a fixed number of variables for the sums.
 */
#include <iostream>
using namespace std;
int main() {
  int n; // Number of force vectors.
  int x, y, z; // Components of a single force vector.
  // sumx, sumy, sumz: Accumulators for the vector sum of all forces.
  int sumx = 0, sumy = 0, sumz = 0;
  
  cin >> n; // Read the number of vectors.

  /**
   * Block Logic: Summation of force vectors.
   * This loop iterates through all 'n' vectors, reading their components
   * and accumulating them into the total sum for each axis.
   */
  while (n--) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }

  /**
   * Block Logic: Equilibrium check.
   * A body is in equilibrium if and only if the net force on it is zero.
   * This means the sum of forces along each independent axis must be zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
