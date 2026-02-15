/**
 * @file 69A.CPP
 * @brief An optimal solution to the "Young Physicist" competitive programming problem.
 * @details The problem is to determine if a body in 3D space is in equilibrium.
 * This occurs if the vector sum of all forces acting on it is the zero vector,
 * meaning the sum of the components along each axis (X, Y, Z) is zero.
 * This solution calculates the sum of the components in a single pass.
 *
 * @section algorithm Algorithm
 * The algorithm efficiently sums the components of all force vectors as they are read:
 * 1. Read the total number of applied force vectors, `n`.
 * 2. Initialize three sum variables, `sum_x`, `sum_y`, and `sum_z`, to zero.
 * 3. Loop `n` times, once for each force vector.
 * 4. In each iteration, read the three components (x, y, z) of a single force vector.
 * 5. Add each component to its corresponding sum variable (`sum_x += x`, etc.).
 * 6. After the loop, the three variables hold the components of the net force vector.
 *    The program then checks if all three sums are zero.
 * 7. If `sum_x`, `sum_y`, and `sum_z` are all zero, it prints "YES".
 * 8. Otherwise, it prints "NO".
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of force vectors, as it uses a single loop to process all vectors.
 * - Space Complexity: O(1), as it uses a fixed number of variables to store the running sums, regardless of the input size.
 */
#include <iostream>
#include <vector>
// The algorithm and iostream are sufficient. Other headers are not strictly needed.
// #include <algorithm>
// #include <string>

using namespace std;

int main() {
  // sum_x, sum_y, sum_z will store the sum of components for each axis.
  int sum_x = 0, sum_y = 0, sum_z = 0;
  // x, y, z are temporary variables for the components of a single vector.
  int x, y, z, n;
  
  // Read the number of force vectors.
  cin >> n;
  
  // Loop through each vector, adding its components to the running sums.
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sum_x += x;
    sum_y += y;
    sum_z += z;
  }
  
  // Check if the net force vector is (0, 0, 0), which signifies equilibrium.
  if (sum_x == 0 && sum_y == 0 && sum_z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  return 0;
}
