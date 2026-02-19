/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem by checking for force equilibrium.
 * @details The program determines if a body is in equilibrium by summing a series of
 * 3D force vectors and checking if the resultant vector is zero. This is based on
 * Newton's First Law.
 *
 * @algorithm The program iterates `n` times, once for each force vector. In each
 * iteration, it reads the x, y, and z components of a vector and adds them to
 * running totals. After all vectors have been processed, it checks if the totals
 * for all three components are zero. If they are, the net force is zero, and the
 * body is in equilibrium.
 *
 * @time_complexity O(N), where N is the number of force vectors, as it involves a
 * single pass through the input data.
 * @space_complexity O(1), as it uses a fixed number of variables for the sums.
 */
#include <iostream>
using namespace std;
int main() {
  // x, y, z will store the sum of the components of the resultant force vector.
  int x = 0, y = 0, z = 0;
  int n;
  cin >> n;
  /**
   * Block Logic: This loop reads 'n' force vectors and accumulates their components.
   * Invariant: At the start of each iteration, x, y, and z hold the sum of components
   *            for all vectors processed so far.
   */
  while (n--) {
    // i, j, k are the components of the current force vector being read.
    int i, j, k;
    cin >> i >> j >> k;
    x += i;
    y += j;
    z += k;
  }
  /**
   * Block Logic: Check for equilibrium.
   * Pre-condition: x, y, and z represent the components of the net force.
   * Equilibrium exists if and only if the net force is the zero vector (all components are zero).
   */
  if (x == 0 && y == 0 && z == 0)
    cout << "YES";
  else
    cout << "NO";
}