/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem.
 * @details This program determines if a body is in equilibrium by calculating the
 * vector sum of all forces acting upon it. A body is in equilibrium if and only
 * if the net force is zero, which means the sum of forces along each coordinate
 * axis (X, Y, and Z) must be zero.
 *
 * @section algorithm Algorithm
 * 1. Read an integer `n`, which represents the number of force vectors.
 * 2. Initialize three variables (`sum_x`, `sum_y`, `sum_z`) to 0. These will
 *    accumulate the total force along the X, Y, and Z axes, respectively.
 * 3. Loop `n` times to process each force vector.
 *    a. In each iteration, read the three components (x, y, z) of a single force.
 *    b. Add each component to its corresponding sum variable.
 * 4. After the loop has processed all vectors, check if all three sum variables
 *    are equal to 0.
 * 5. If `sum_x`, `sum_y`, and `sum_z` are all 0, it means the net force is zero,
 *    and the program prints "YES".
 * 6. Otherwise, the body is not in equilibrium, and the program prints "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of forces, because the program
 *   iterates through each force vector exactly once.
 * - Space Complexity: O(1), as it uses a fixed amount of memory for the sum
 *   variables, regardless of the number of forces.
 */
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Main function to check for force equilibrium.
 * @return 0 on successful execution.
 */
int main() {
  // sum_x, sum_y, sum_z will store the net force components.
  int sum_x = 0, sum_y = 0, sum_z = 0;
  int x_comp, y_comp, z_comp, n;
  
  // Read the number of force vectors.
  cin >> n;
  
  // Loop through each force vector.
  for (int i = 0; i < n; i++) {
    // Read the components of the current force.
    cin >> x_comp >> y_comp >> z_comp;
    // Add them to the running totals.
    sum_x += x_comp;
    sum_y += y_comp;
    sum_z += z_comp;
  }
  
  // Check if the body is in equilibrium.
  if (sum_x == 0 && sum_y == 0 && sum_z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  return 0;
}
