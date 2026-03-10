/**
 * @file 69A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Young Physicist").
 * This program determines if a body is in equilibrium by checking if the vector
 * sum of all applied forces is zero.
 *
 * @algorithm Summation of Vector Components. The principle of equilibrium in
 * physics states that an object is in equilibrium if and only if the net force
 * acting upon it is zero. For 3D vectors, this means the sum of the x, y, and z
 * components must all individually be zero. The program iterates through the
 * force vectors, maintains a running sum for each component, and checks this
 * condition at the end.
 *
 * @time_complexity O(N), where N is the number of force vectors, due to a single pass.
 * @space_complexity O(1), as it only stores the running sums.
 */
#include <iostream>
using namespace std;

// x, y, z store the components of a single input vector.
// x1, y1, z1 store the sum of all x, y, and z components, respectively.
int n, x, y, z, x1 = 0, y1 = 0, z1 = 0;

int main() {
  cin >> n;
  /**
   * @block
   * @brief Reads each force vector and accumulates the component sums.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    x1 += x;
    y1 += y;
    z1 += z;
  }

  /**
   * @block
   * @brief Checks the equilibrium condition.
   * If the sum of forces on each axis is zero, the net force is zero.
   */
  if (x1 == 0 && y1 == 0 && z1 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}