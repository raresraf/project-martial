/**
 * @file 69A.CPP
 * @brief Determines if a body is in equilibrium given a set of force vectors.
 * @details This program solves the "Young Physicist" problem from Codeforces.
 * A body is in equilibrium if and only if the vector sum of all forces acting
 * upon it is zero. The program calculates this vector sum and checks if it is
 * the zero vector.
 *
 * @author Thiago Ribeiro
 * @date 6/27/17
 *
 * @see https://codeforces.com/problemset/problem/69/A
 *
 * Algorithm:
 * 1. Initialize the components of the net force vector (x, y, z) to zero.
 * 2. Read the number of applied force vectors, n.
 * 3. Loop n times:
 *    a. Read the x, y, and z components of a single force vector.
 *    b. Add these components to the corresponding components of the net force vector.
 * 4. After the loop, check if all three components of the net force are zero.
 * 5. If `x == 0`, `y == 0`, and `z == 0`, the body is in equilibrium ("YES").
 *    Otherwise, it is not ("NO").
 *
 * Time Complexity: O(N), where N is the number of force vectors, as the program
 * iterates through the input vectors once.
 * Space Complexity: O(1), as a fixed amount of memory is used regardless of N.
 */
#include <iostream>
#include <cstdio> // For scanf

using namespace std;

int main() {
  // n: the number of force vectors.
  int n;
  // x, y, z: store the sum of the components of all force vectors.
  int x = 0, y = 0, z = 0;
  // a, b, c: store the components of a single input vector.
  int a, b, c;

  cin >> n;

  /**
   * @brief Main loop to sum the force vectors.
   *
   * Pre-condition: n is the number of vectors to process.
   * Invariant: At the end of each iteration, x, y, and z hold the vector
   *            sum of all forces read so far.
   */
  while (n--) {
    // Note: mixing cin/cout with scanf/printf can be problematic without
    // disabling I/O synchronization, but works here due to simple usage.
    scanf("%d %d %d", &a, &b, &c);
    x += a;
    y += b;
    z += c;
  }

  // A body is in equilibrium if the net force is zero in all dimensions.
  if (x == 0 && y == 0 && z == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
