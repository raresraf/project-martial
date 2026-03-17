/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 69A "Young Physicist".
 *
 * @details
 * This program determines if a body is in equilibrium under the influence of several
 * force vectors. A body is in equilibrium if and only if the vector sum of all
 * forces acting on it is the zero vector.
 *
 * @algorithm
 * The algorithm is based on the principle of vector addition.
 * 1. Initialize sums for each vector component (x, y, z) to zero.
 * 2. Read the number of force vectors, `n`.
 * 3. Loop `n` times, reading the x, y, and z components of each vector.
 * 4. Add each component to its corresponding total sum.
 * 5. After processing all vectors, check if the sums of the x, y, and z components
 *    are all equal to zero.
 * 6. If `sumX == 0`, `sumY == 0`, and `sumZ == 0`, the body is in equilibrium ("YES").
 *    Otherwise, it is not ("NO").
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of force vectors, as the program
 *   iterates through the input once.
 * - Space Complexity: O(1), as it only uses a few variables to store the component
 *   sums, regardless of the number of vectors.
 */

#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int n; // Number of force vectors.
  int x, y, z; // Components of a single force vector.
  // Accumulators for the sum of components along each axis.
  int sumX = 0, sumY = 0, sumZ = 0;
  
  std::cin >> n;
  
  /**
   * @brief Read each vector and add its components to the total sums.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> x >> y >> z;
    sumX += x;
    sumY += y;
    sumZ += z;
  }
  
  /**
   * @brief Check for equilibrium.
   * If the net force vector is (0, 0, 0), the body is in equilibrium.
   */
  if (sumX == 0 && sumY == 0 && sumZ == 0)
    std::cout << "YES";
  else
    std::cout << "NO";
    
  return 0;
}
