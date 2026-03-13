/**
 * @file 69A.CPP
 * @brief Solution for CodeForces Problem 69A - "Young Physicist".
 * 
 * @details
 * The problem asks to determine if a body is in equilibrium given a set of force vectors
 * applied to it. A body is in equilibrium if the vector sum of all forces is zero.
 * This means the sum of the x-components, the sum of the y-components, and the sum
 * of the z-components must all be zero.
 *
 * @algorithm Vector Summation
 * 1. Read the number of force vectors, `n`.
 * 2. Initialize three variables (`a`, `b`, `c`) to store the sum of x, y, and z components, respectively.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add them to the running sums `a`, `b`, and `c`.
 * 4. After the loop, check if `a`, `b`, and `c` are all zero.
 * 5. If they are, print "YES"; otherwise, print "NO".
 *
 * @time_complexity O(N), where N is the number of force vectors.
 * @space_complexity O(1), as the storage used is constant regardless of N.
 *
 * @note The declaration `int arr[101][101];` is unused in this program.
 */
#include <iostream>
using namespace std;
typedef long long ll;

int main() {
  int n; // Number of force vectors.
  cin >> n;

  ll a = 0, b = 0, c = 0; // Sum of x, y, and z components respectively.
  ll a1, a2, a3;           // Components of a single force vector.

  /**
   * @brief Main loop to read vectors and accumulate their component sums.
   * Invariant: `a`, `b`, and `c` hold the vector sum of all forces processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> a1 >> a2 >> a3;
    a += a1;
    b += a2;
    c += a3;
  }

  /**
   * @brief Check for equilibrium.
   * If the sum of all x, y, and z components is zero, the net force is zero,
   * and the body is in equilibrium.
   */
  if (a == 0 && b == 0 && c == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
