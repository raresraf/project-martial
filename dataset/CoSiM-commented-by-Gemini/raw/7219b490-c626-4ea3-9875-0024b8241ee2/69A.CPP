/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 *
 * This program checks for static equilibrium in a system of 3D force vectors.
 * It reads 'n' vectors and determines if their vector sum is zero. If the sum
 * of forces along each of the x, y, and z axes is zero, the system is in
 * equilibrium, and the program outputs "YES". Otherwise, it outputs "NO".
 *
 * @algorithm Vector component summation.
 * @time_complexity O(N), where N is the number of vectors. Although there is a
 *                  nested loop, the outer loop runs a constant 3 times (for x, y, z),
 *                  and the inner loop runs N times.
 * @space_complexity O(N) to store the N vectors in a 2D array.
 */
#include <iostream>
using namespace std;

int main() {
  int n, i, j;
  cin >> n;
  // A variable-length array (VLA) is used here, which is a non-standard C++ extension.
  int arr[n][3];
  /**
   * @brief Reads the components of the N vectors.
   * This nested loop populates the 2D array with the x, y, and z components
   * of each of the 'n' vectors.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }
  bool ans = true;
  /**
   * @brief Calculates the sum for each component and checks for equilibrium.
   * This loop iterates through each of the three dimensions (j = 0, 1, 2).
   * For each dimension, it calculates the sum of all vectors' components.
   * If any of these sums is not zero, the system is not in equilibrium.
   */
  for (j = 0; j < 3; j++) {
    int sum = 0;
    for (i = 0; i < n; i++) {
      sum += arr[i][j];
    }
    if (sum != 0) {
      ans = false;
    }
  }

  // Print the final result based on the equilibrium check.
  if (ans) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}