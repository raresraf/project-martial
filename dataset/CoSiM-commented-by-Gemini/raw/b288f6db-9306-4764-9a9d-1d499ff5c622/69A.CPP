/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - "Young Physicist".
 * @details This program determines if a body is in equilibrium under the influence of several forces.
 *          Equilibrium is achieved if the vector sum of all forces is zero. This means the sum
 *          of the x, y, and z components of all force vectors must each be zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize sums for each coordinate (x, y, z) to zero.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add these components to their respective running sums.
 * 4. After the loop, check if all three sums (x, y, and z) are zero.
 * 5. If all sums are zero, print "YES".
 * 6. Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1).
 */

/*------------***********-----------
        Mostafijur Rahman Kajol
         CSE_15, JUST.
    ------********-------------------
    */

#include <algorithm>
#include <bits/stdc++.h>
#include <istream>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace std;

int main() {
  int n, a, b, c;
  // Initialize sums of vector components for each axis.
  int x = 0, y = 0, z = 0;
  
  cin >> n;
  
  /**
   * @brief Main loop to read `n` force vectors and sum their components.
   * Invariant: At the end of each iteration, x, y, and z hold the sum of components
   * for all vectors read so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    x += a; // Summing x-components
    y += b; // Summing y-components
    z += c; // Summing z-components
  }

  /**
   * @brief Check for equilibrium condition.
   * The body is in equilibrium if the net force vector is (0, 0, 0).
   */
  if (x == 0 && y == 0 && z == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
