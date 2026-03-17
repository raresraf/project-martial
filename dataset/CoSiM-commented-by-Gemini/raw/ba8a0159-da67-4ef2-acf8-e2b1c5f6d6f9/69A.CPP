/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem.
 *
 * @details This program determines if a body is in equilibrium under the influence of several forces.
 * It reads 'n' force vectors, each with three spatial components (x, y, z).
 * According to Newton's first law, a body is in equilibrium if and only if the vector
 * sum of all forces acting on it is zero. The program calculates the sum of all x, y, and z
 * components respectively and checks if all three sums are zero.
 *
 * @section
 * Algorithm:
 * 1. Read the number of force vectors, n.
 * 2. Initialize sum of x, y, and z components (sumx, sumy, sumz) to 0.
 * 3. Loop n times:
 *    a. Read the x, y, and z components of a vector.
 *    b. Add them to their respective sums.
 * 4. After the loop, check if sumx, sumy, and sumz are all zero.
 * 5. If they are, print "YES"; otherwise, print "NO".
 *
 * Time Complexity: O(n), because the program iterates through the n vectors once.
 * Space Complexity: O(1), as memory usage is constant and does not depend on n.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x, y, z, sumx = 0, sumy = 0, sumz = 0;
  cin >> n;
  /**
   * @brief This loop iterates through each of the 'n' force vectors, accumulating their components.
   * Invariant: At the start of each iteration, sumx, sumy, and sumz hold the component-wise
   * sum of all vectors processed so far.
   */
  while (n--) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  /**
   * @brief This block checks the equilibrium condition.
   * Pre-condition: sumx, sumy, and sumz hold the total sum of all vector components.
   * Logic: If the net vector is (0, 0, 0), the body is in equilibrium.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
}
