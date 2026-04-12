/**
 * @file 69A.CPP
 * @brief Static equilibrium analysis of 3D force vectors.
 * 
 * Logic: Checks if a physical body is in equilibrium by summing the components of 
 * N applied force vectors. Equilibrium is achieved if the resultant vector 
 * magnitude is zero (sum of x, y, and z components are each zero).
 * 
 * Algorithm: Component-wise linear accumulation.
 * Time Complexity: $O(N)$ where N is the number of force vectors.
 * Space Complexity: $O(1)$ auxiliary space (ignoring the unused 2D array).
 */

#include <iostream>
using namespace std;
typedef long long ll;

int main() {
  // Note: Static allocation of a 101x101 matrix is present but unused in the core logic.
  int arr[101][101];
  int n;
  cin >> n;

  // Accumulators for the resultant vector components in Cartesian coordinates.
  ll a = 0, b = 0, c = 0;
  ll a1, a2, a3;

  /**
   * Block Logic: Iterative vector summation.
   * Invariant: At the end of iteration i, a, b, and c represent the sum of the first i vectors.
   */
  for (int i = 0; i < n; i++) {
    // Reads components: a1 (x), a2 (y), a3 (z).
    cin >> a1 >> a2 >> a3;
    // Functional Utility: Accumulates the components into the resultant vector.
    a += a1;
    b += a2;
    c += a3;
  }

  /**
   * Functional Utility: Evaluates the zero-resultant equilibrium condition.
   */
  if (a == 0 && b == 0 && c == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  return 0;
}
