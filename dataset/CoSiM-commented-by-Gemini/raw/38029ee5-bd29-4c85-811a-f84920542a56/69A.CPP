/**
 * @file 69A.CPP
 * @brief A space-optimized solution to the "Young Physicist" problem (Codeforces 69A),
 *        checking for force equilibrium.
 *
 * @details This program determines if a set of 3D force vectors results in a net force of zero.
 * It uses a memory-efficient approach by not storing the individual vectors. A body is in
 * equilibrium if the vector sum of all forces is zero, which means the sum of components
 * along each axis must be zero.
 *
 * @algorithm The program reads the number of vectors, `n`. It then iterates `n` times,
 * reading one vector's components (x, y, z) in each iteration. These components are
 * immediately added to running totals for each axis. This avoids allocating arrays to store
 * all the vectors. After the loop, it checks if the totals for all three axes are zero.
 * If so, the system is in equilibrium ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of vectors, for a single pass through the inputs.
 * Space Complexity: O(1), as the program only requires a constant amount of extra space
 * for the sum variables, regardless of the number of vectors.
 */
#include <iostream>

using namespace std;

int main() {
  int sumx = 0, sumy = 0, sumz = 0, n;
  cin >> n;
  /**
   * @brief Block Logic: Reads `n` vectors and accumulates their components in-place.
   * @invariant After `i` iterations, `sumx`, `sumy`, and `sumz` hold the component-wise
   * sum of the first `i` vectors.
   */
  for (int i = 0; i < n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  
  /**
   * @brief Block Logic: Checks for the equilibrium condition.
   * The condition for equilibrium is that the net force vector is (0, 0, 0).
   */
  if ((sumx == 0) && (sumy == 0) && (sumz == 0)) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;
  return 0;
}