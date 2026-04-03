/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (Problem 69A).
 * @see https://codeforces.com/problemset/problem/69/A
 *
 * @description
 * The problem provides a set of n 3D force vectors. The task is to determine if
 * a body subjected to these forces is in equilibrium.
 *
 * @algorithm
 * A body is in equilibrium if and only if the net force acting upon it is zero.
 * The net force is the vector sum of all individual forces. For the net force vector
 * to be zero, the sum of its components along each axis (x, y, and z) must be zero.
 *
 * The algorithm is as follows:
 * 1. Initialize three variables, `sumx`, `sumy`, and `sumz`, to zero.
 * 2. Read the number of vectors, n.
 * 3. Loop n times, reading the x, y, and z components of each force vector.
 * 4. Add the components of each vector to the corresponding sum variables.
 * 5. After the loop, check if `sumx`, `sumy`, and `sumz` are all zero.
 * 6. If they are, the body is in equilibrium ("YES"). Otherwise, it is not ("NO").
 *
 * @time_complexity O(N), where N is the number of force vectors, due to the single
 *                  loop required to read and sum the vector components.
 * @space_complexity O(1), as the program only requires a few variables to store the
 *                   component sums, regardless of the number of vectors.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
main() {
  int n;
  cin >> n;

  // Variables to store the sum of components along each axis.
  long long sumx = 0;
  long long sumy = 0;
  long long sumz = 0;

  int x, y, z;

  /**
   * @brief Block Logic: Read all force vectors and accumulate component sums.
   * This loop iterates n times, once for each force vector, adding the
   * x, y, and z components to their respective totals.
   */
  for (int i = 0; i < n; i++) {
    cin >> x;
    sumx += x;
    cin >> y;
    sumy += y;
    cin >> z;
    sumz = sumz + z;
  }

  /**
   * @brief Block Logic: Check for equilibrium.
   * If the sum of forces along each axis is zero, the net force is zero,
   * and the body is in equilibrium.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
}
