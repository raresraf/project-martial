/**
 * @file 69A.CPP
 * @brief Solution for the "Young Physicist" problem (Codeforces 69A).
 * @details This program determines if a body is in equilibrium under the influence
 * of multiple force vectors. A body is in equilibrium if the vector sum of all
 * forces acting upon it is the zero vector.
 *
 * Algorithm: Vector Component Summation
 * The principle of equilibrium states that the sum of all x-components, y-components,
 * and z-components of the forces must each be zero.
 * The program iterates through all the given force vectors, accumulating the sum for
 * each of the three (x, y, z) components separately. After processing all vectors,
 * it checks if all three sums are equal to zero.
 *
 * Time Complexity: O(N), where N is the number of force vectors, as the program
 * iterates through each vector once.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */

#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  // n: the number of force vectors.
  int n;
  cin >> n;
  
  // p, q, r: store the x, y, and z components of a single force vector read in each iteration.
  int p = 0, q = 0, r = 0;
  // a, b, c: store the sum of the x, y, and z components of all vectors, respectively.
  int a = 0, b = 0, c = 0;
  
  /**
   * Block Logic: Read each force vector and accumulate the component sums.
   * Pre-condition: 'n' is the number of vectors to process.
   * Invariant: At the end of each iteration, 'a', 'b', and 'c' hold the
   * component-wise sum of the first 'i' vectors.
   */
  for (int i = 0; i < n; i++) {
    cin >> p >> q >> r;
    a += p;
    b += q;
    c += r;
  }
  
  // Block Logic: Check for equilibrium.
  // The body is in equilibrium if and only if the sum of components along
  // each axis is zero.
  if ((a == 0) && (b == 0) && (c == 0))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
    
  return 0;
}
