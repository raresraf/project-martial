/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem.
 * @details This program determines if a body is in equilibrium under the influence of several forces.
 * According to Newton's First Law, a body is in equilibrium if and only if the vector sum of all
 * forces acting on it is zero. This means the sum of the x-components, y-components, and
 * z-components must all independently be zero.
 *
 * @section algorithm Algorithm
 * 1. Read the number of forces, `n`.
 * 2. Initialize three variables, `sum_x`, `sum_y`, and `sum_z`, to 0. These will store the
 *    net force components along the X, Y, and Z axes.
 * 3. Loop `n` times, once for each force vector.
 *    a. In each iteration, read the three components of a force vector (x, y, z).
 *    b. Add each component to its corresponding sum variable (`sum_x += x`, `sum_y += y`, `sum_z += z`).
 * 4. After the loop has processed all the force vectors, check if `sum_x`, `sum_y`, and `sum_z` are all equal to 0.
 * 5. If all three sums are 0, the body is in equilibrium, so the program prints "YES".
 * 6. Otherwise, it prints "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of forces, as the program iterates through each force vector once.
 * - Space Complexity: O(1), using a fixed amount of memory for the sum variables.
 */
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Main function to check for equilibrium.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  cin >> n;
  
  int sum_x = 0, sum_y = 0, sum_z = 0;
  
  // Loop through each of the 'n' force vectors.
  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    // Add the components of the current force to the total sums.
    sum_x += x;
    sum_y += y;
    sum_z += z;
  }
  
  // Check if the net force vector is (0, 0, 0).
  if (sum_x == 0 && sum_y == 0 && sum_z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  return 0;
}