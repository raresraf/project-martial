/**
 * @file 69A.CPP
 * @brief Static equilibrium analysis of a particle subject to multiple force vectors in 3D space.
 * 
 * Logic: Calculates the vector sum of N force vectors in a Cartesian coordinate system. 
 * A system is in static equilibrium (YES) if and only if the resultant force 
 * vector is the null vector [0, 0, 0].
 * 
 * Algorithm: Component-wise linear accumulation.
 * Time Complexity: $O(N)$ where N is the number of force vectors.
 * Space Complexity: $O(1)$ constant auxiliary space.
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
  // Accumulators for the resultant force components (x, y, z).
  int x = 0, y = 0, z = 0;
  
  // Functional Utility: Reads the total number of applied force vectors.
  if (!(cin >> n)) return 0;

  /**
   * Block Logic: Vector summation pass.
   * Invariant: At any step i, (x, y, z) represents the net force of the first i vectors.
   */
  for (int i = 0; i < n; i++) {
    // Reads components of the current force vector.
    cin >> a >> b >> c;
    // Functional Utility: Component-wise summation.
    x += a;
    y += b;
    z += c;
  }

  /**
   * Functional Utility: Final equilibrium condition check.
   * Logic: Returns "YES" if all orthogonal components sum to zero.
   */
  if (x == 0 && y == 0 && z == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
