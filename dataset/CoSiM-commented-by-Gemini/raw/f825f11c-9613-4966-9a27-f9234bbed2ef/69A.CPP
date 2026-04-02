/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem (Codeforces 69A).
 *
 * Problem Statement:
 * Determine if a body is in equilibrium given a set of force vectors acting on it.
 * A body is in equilibrium if the vector sum of all forces is zero. This means
 * the sum of the x-components, the sum of the y-components, and the sum of the
 * z-components must all independently be zero.
 *
 * Algorithm:
 * The program reads the number of force vectors, 'n'. It then iterates 'n'
 * times, reading the x, y, and z components of each force vector. It maintains
 * three variables, `sumx`, `sumy`, and `sumz`, to accumulate the sum of the
 * components along each respective axis. After reading all vectors, it checks
 * if all three sums are zero. If they are, the body is in equilibrium ("YES"),
 * otherwise it is not ("NO").
 *
 * Note on Implementation:
 * The code declares arrays `x`, `y`, and `z` and stores the vector components
 * in them. However, these arrays are not used for any computation after their
 * values are added to the sums, making their storage redundant for the final
 * calculation.
 *
 * Time Complexity: O(n), for iterating through the 'n' force vectors.
 * Space Complexity: O(1), as the space used does not scale with 'n' (the arrays
 * have a fixed size).
 */
#include <iostream>
#include <string>

using namespace std;

int main() {

  int n;
  // Unnecessary arrays, as their values are used only to update the sums.
  int x[100], y[100], z[100];
  cin >> n;
  
  // Variables to store the sum of components for each axis.
  int sumx = 0;
  int sumy = 0;
  int sumz = 0;
  int i = 0;

  /**
   * Block Logic: Read all 'n' vectors and compute the sum of their components.
   * Pre-condition: 'n' is the number of vectors to be read.
   * Invariant: sumx, sumy, sumz hold the running total of components.
   */
  while (n--) {
    cin >> x[i] >> y[i] >> z[i];
    sumx += x[i];
    sumy += y[i];
    sumz += z[i];
    i++;
  }

  /**
   * Block Logic: Check for equilibrium condition.
   * The body is in equilibrium if the net force along each axis is zero.
   */
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
}
