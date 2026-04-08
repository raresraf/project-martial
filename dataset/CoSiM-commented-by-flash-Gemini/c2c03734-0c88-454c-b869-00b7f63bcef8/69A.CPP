/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (Problem 69A).
 * @details This program checks if a body is in equilibrium by summing up all
 * the force vectors acting on it. For the body to be in equilibrium, the net
 * force along each of the three axes (x, y, z) must be zero.
 *
 * @algorithm
 * 1. Read the number of applied force vectors, `n`.
 * 2. Initialize three variables (`a`, `b`, `c`) to store the sum of forces
 *    along the x, y, and z axes, respectively.
 * 3. Loop `n` times, once for each force vector.
 * 4. In each iteration, read the three components of the vector and add them
 *    to the corresponding sum variables.
 * 5. After the loop, if the sums for all three axes are zero, the net force
 *    is the zero vector, and the body is in equilibrium. Print "YES".
 * 6. Otherwise, the body is not in equilibrium. Print "NO".
 *
 * @time_complexity O(n), where `n` is the number of force vectors.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int p = 0, q = 0, r = 0; // Temporary variables for a single vector's components
  int a = 0, b = 0, c = 0; // Sum of components along x, y, and z axes

  /**
   * @block_logic Loop to read `n` force vectors and calculate the total force on each axis.
   */
  for (int i = 0; i < n; i++) {
    cin >> p >> q >> r;
    a += p;
    b += q;
    c += r;
  }

  // Check if the body is in equilibrium.
  if ((a == 0) && (b == 0) && (c == 0))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}