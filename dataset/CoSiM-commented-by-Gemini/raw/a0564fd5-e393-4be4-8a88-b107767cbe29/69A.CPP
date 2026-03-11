/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem (Codeforces 69A).
 * @details The problem asks to determine if a body is in equilibrium under the
 *          action of multiple forces. A body is in equilibrium if and only if
 *          the vector sum of all forces acting on it is the zero vector.
 *
 * @algorithm To check for equilibrium, we need to verify that the sum of the
 *            components of all force vectors is zero for each coordinate axis
 *            (x, y, and z).
 *            1. Read all the force vectors.
 *            2. Calculate the sum of all x-components, the sum of all y-components,
 *               and the sum of all z-components.
 *            3. If all three sums are zero, the body is in equilibrium. Otherwise, it is not.
 *
 * @time_complexity O(N), where N is the number of force vectors, as it requires
 *                  iterating through the vectors to sum up the components.
 * @space_complexity O(N) to store the N force vectors.
 */
#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  int t;
  cin >> t;
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  int arr[t][3];

  // Read all the force vectors' components.
  for (int i = 0; i < t; i++)
    for (int j = 0; j < 3; j++)
      cin >> arr[i][j];

  vector<int> res;
  /**
   * @brief Calculate the sum of components for each of the three dimensions.
   * j=0 corresponds to x-components, j=1 to y-components, and j=2 to z-components.
   */
  for (int j = 0; j < 3; j++) {
    int sum = 0;
    for (int i = 0; i < t; i++) {
      sum += arr[i][j];
    }
    res.push_back(sum);
  }

  /**
   * @brief Check if the total force vector is zero.
   * This is true if the sum of components along each axis is zero.
   */
  if (res[0] == 0 && res[1] == 0 && res[2] == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
