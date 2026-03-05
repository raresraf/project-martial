/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A, "Young Physicist".
 * @details This program checks if a body is in equilibrium given a set of `n`
 * 3D force vectors. Equilibrium is achieved if the vector sum of all forces is zero.
 *
 * Algorithm:
 * The code uses a two-pass approach.
 * 1. It first reads all `n` force vectors and stores their x, y, and z
 *    components into three separate arrays. Note that it uses Variable Length
 *    Arrays (VLAs), which are a non-standard C++ extension.
 * 2. In a second pass, it iterates through the arrays, summing the components
 *    for each dimension to find the resultant force vector.
 * 3. Finally, it checks if all three components of the resultant vector are
 *    zero. If so, the body is in equilibrium ("YES"), otherwise it is not ("NO").
 *
 * This approach is correct but less optimal than a single-pass solution, as it
 * uses O(N) extra space for the arrays and requires two separate loops.
 *
 * Time Complexity: O(N), for two separate linear passes over the data.
 * Space Complexity: O(N), due to storing all vector components in arrays.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n;
  cin >> n;

  // sum1, sum2, sum3 will store the net force on each axis.
  // The x, y, z arrays use a non-standard C++ extension (Variable Length Arrays).
  int sum1 = 0, sum2 = 0, sum3 = 0, x[n + 10], y[n + 10], z[n + 10];

  /**
   * @brief First pass: Read all `n` force vectors and store them in arrays.
   */
  for (int i = 0; i < n; i++)
    cin >> x[i] >> y[i] >> z[i];
  
  /**
   * @brief Second pass: Calculate the sum of forces for each dimension.
   */
  for (int i = 0; i < n; i++) {
    sum1 += x[i];
    sum2 += y[i];
    sum3 += z[i];
  }

  /**
   * @brief Check for equilibrium. The net force on each axis must be zero.
   */
  if (sum1 == 0 && sum2 == 0 && sum3 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
