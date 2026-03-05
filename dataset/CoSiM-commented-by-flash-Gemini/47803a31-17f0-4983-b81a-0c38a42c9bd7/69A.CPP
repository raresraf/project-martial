/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem (Codeforces 69A).
 *
 * This program determines if a body is in equilibrium given 'n' forces acting
 * upon it. A body is in equilibrium if the sum of all force vectors (x, y, z components)
 * is zero. The program reads 'n' force vectors and checks if the net force in
 * each dimension (X, Y, Z) is zero.
 *
 * Algorithm: Vector summation and conditional check. It sums up the components
 * of all force vectors and then verifies if all three resultant sums are zero.
 * Time Complexity: O(N) where N is the number of force vectors.
 * Space Complexity: O(1) as only a few variables are used to store the sums.
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function to check if a body is in equilibrium.
 *
 * Reads the number of force vectors 'n', then reads 'n' sets of (x, y, z)
 * components. It calculates the sum of each component and determines if
 * all three sums are zero, indicating equilibrium.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare integers for the sums of x, y, and z components, and for individual components.
  int sumx, sumy, sumz, xi, yi, zi, n;
  // Initialize sum variables to zero.
  sumx = 0;
  sumy = 0;
  sumz = 0;
  // Read the number of force vectors 'n' from standard input.
  cin >> n;
  /**
   * @brief Block Logic: Reads force vectors and accumulates component sums.
   *
   * Iterates 'n' times, reading the x, y, and z components of each force
   * vector and adding them to their respective sum variables.
   * Pre-condition: 'n' specifies the total number of force vectors.
   * Invariant: 'sumx', 'sumy', 'sumz' accumulate the total force in each dimension.
   */
  for (int i = 0; i < n; i++) {
    // Read the x, y, and z components of the current force vector.
    cin >> xi >> yi >> zi;
    // Add components to their respective sums.
    sumx = sumx + xi;
    sumy = sumy + yi;
    sumz = sumz + zi;
  }
  // Pre-condition: 'sumx', 'sumy', 'sumz' hold the total force components.
  // Block Logic: Checks if the net force in all three dimensions is zero.
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES"; // The body is in equilibrium.
  else
    cout << "NO";  // The body is not in equilibrium.
  return 0;
}
