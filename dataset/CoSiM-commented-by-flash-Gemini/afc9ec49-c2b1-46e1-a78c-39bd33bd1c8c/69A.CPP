/**
 * @file 69A.CPP
 * @brief Static equilibrium analysis of a particle in 3D Euclidean space.
 * 
 * Logic: Calculates the vector sum of N force vectors. The system is in 
 * equilibrium (YES) if the magnitude of the resultant vector is zero, 
 * which requires each component (x, y, z) to independently sum to zero.
 * 
 * Algorithm: Component-wise accumulation.
 * Time Complexity: $O(N)$ where N is the number of force vectors.
 * Space Complexity: $O(1)$ auxiliary space (ignoring the unused 2D array).
 */

#include <iostream>
using namespace std;
typedef long long ll;

int main() {
  /**
   * Note: Static allocation of a 101x101 matrix is present in the source but 
   * is not utilized in the core equilibrium logic.
   */
  int arr[101][101];
  int n;
  // Reads total number of applied force vectors.
  if (!(cin >> n)) return 0;

  // Resultant vector components (x, y, z).
  ll res_x = 0, res_y = 0, res_z = 0;
  ll a1, a2, a3;

  /**
   * Block Logic: Main vector addition loop.
   * Invariant: Accumulates the net force along each orthogonal axis.
   */
  for (int i = 0; i < n; i++) {
    // Reads components: a1 (x), a2 (y), a3 (z).
    cin >> a1 >> a2 >> a3;
    // Functional Utility: Linearly sums individual components.
    res_x += a1;
    res_y += a2;
    res_z += a3;
  }

  /**
   * Functional Utility: Resultant vector evaluation.
   * Equilibrium Condition: All components of the net force must be zero.
   */
  if (res_x == 0 && res_y == 0 && res_z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  return 0;
}
