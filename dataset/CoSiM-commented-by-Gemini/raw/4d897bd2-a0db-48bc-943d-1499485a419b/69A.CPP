/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 * @details This program determines if a body is in equilibrium under the influence
 * of several force vectors. A body is in equilibrium if the vector sum of all
 * forces acting on it is the zero vector (i.e., the sum of the x, y, and z
 * components are all individually zero).
 *
 * @algorithm
 * 1. Read the number of force vectors, `n`.
 * 2. For each of the `n` vectors, read its x, y, and z components.
 * 3. Maintain three separate sums for the x, y, and z components of all vectors.
 * 4. Iterate through all vectors, adding each component to its corresponding sum.
 * 5. After summing all vectors, check if the x-sum, y-sum, and z-sum are all zero.
 * 6. If all sums are zero, the body is in equilibrium ("YES"). Otherwise, it is
 *    not ("NO").
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of vectors, as the program
 * iterates through the vectors twice (once for input, once for summation).
 * Space Complexity: O(N) to store the N vectors.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(void) {
  int numberOfVectors;
  cin >> numberOfVectors;

  /**
   * @note The code uses a Variable-Length Array (VLA), `arr[numberOfVectors][3]`.
   * This is a C99 feature and not part of the standard C++. While supported by
   * some compilers like g++ as an extension, using `std::vector` would be more
   * portable and idiomatic in C++.
   */
  int arr[numberOfVectors][3];

  /**
   * @block
   * @brief Read all vector components into the 2D array.
   */
  for (int i = 0; i < numberOfVectors; i++)
    cin >> arr[i][0] >> arr[i][1] >> arr[i][2];

  // Initialize sums for each component (x, y, z).
  int x_sum = 0, y_sum = 0, z_sum = 0;

  /**
   * @block
   * @brief Calculate the sum of components for all vectors.
   */
  for (int i = 0; i < numberOfVectors; i++) {
    x_sum += arr[i][0];
    y_sum += arr[i][1];
    z_sum += arr[i][2];
  }

  /**
   * @block
   * @brief Check for equilibrium.
   * @details If the sum of forces along each axis is zero, the net force is the
   * zero vector, and the body is in equilibrium.
   */
  if (x_sum == 0 && y_sum == 0 && z_sum == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
