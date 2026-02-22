/**
 * @file 69A.CPP
 * @brief An optimal solution for the "Young Physicist" problem (Codeforces 69A).
 *
 * @algorithm
 * The problem is to determine if a body is in equilibrium given `n` force
 * vectors. A body is in equilibrium if the vector sum of all forces is zero,
 * which means the sums of the x, y, and z components must all be zero.
 *
 * This solution efficiently calculates the net force in a single pass without
 * storing all the vectors, resulting in O(1) space complexity.
 * 1. Initialize sum variables for each axis (`sumx`, `sumy`, `sumz`) to 0.
 * 2. Read the number of force vectors, `n`.
 * 3. Loop `n` times, reading the (x, y, z) components of each vector.
 * 4. Add each component to its corresponding sum variable.
 * 5. After the loop, if `sumx`, `sumy`, and `sumz` are all 0, the net force is
 *    zero, and the body is in equilibrium ("YES"). Otherwise, it is not ("NO").
 *
 * The code uses a macro `do_not_sync` to accelerate C++ I/O operations, a
 * common practice in competitive programming.
 *
 * @complexity
 * - Time Complexity: O(N), for a single pass through the N force vectors.
 * - Space Complexity: O(1), as only a few variables are needed to store the sums.
 */
#include "bits/stdc++.h"
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define FIO                                                                    \
  freopen("input.txt", "r", stdin);                                            \
  freopen("output.txt", "w", stdout);

using namespace std;

int main(void) {
  do_not_sync;
  // FIO
  int n;
  cin >> n;
  // Initialize sums of force components for each axis to zero.
  int sumx = 0, sumy = 0, sumz = 0;

  /**
   * @brief Loop to read each force vector and add its components to the sums.
   */
  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }

  /**
   * @brief Check for equilibrium. The body is in equilibrium if the net force
   * along each axis is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
