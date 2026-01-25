/*
 * 69A - Young Physicist.cpp
 * http://codeforces.com/problemset/problem/69/A
 *  Created on: Oct 28, 2017
 *      Author: athena
 * 31849100 	69A - 23 	GNU C++14 	Accepted 	62 ms 	8 KB
 * 2017-10-28 22:26:28 	2017-10-28 22:26:28
 */

/**
 * @file 69A.CPP
 * @brief Determines if a body is in equilibrium given a set of force vectors.
 * @note Problem: http://codeforces.com/problemset/problem/69/A
 *
 * This program reads a set of 3D force vectors and determines if their vector sum is zero.
 * If the sum of all x, y, and z components is zero, the body is in equilibrium, and the
 * program outputs "YES". Otherwise, it outputs "NO".
 *
 * @b Algorithm: The program iterates through N force vectors, accumulating the sum of their
 * x, y, and z components separately. After processing all vectors, it checks if all three
 * component sums are zero.
 *
 * @b Time_Complexity: O(N), where N is the number of force vectors, as it requires a single
 * pass to sum the components.
 * @b Space_Complexity: O(1), as it only stores the running sums of the vector components,
 * regardless of the number of vectors.
 */

#include <iostream>
using namespace std;

int main() {
  int n;
  // Read the number of force vectors.
  cin >> n;

  int x, y, z;
  int countX(0), countY(0), countZ(0);

  /**
   * @brief This loop reads each force vector and accumulates the component sums.
   *
   * Pre-condition: 'n' is the number of vectors to process.
   * Invariant: At the end of each iteration, `countX`, `countY`, and `countZ` hold the
   * cumulative sum of the force components for the vectors processed so far.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    countX += x;
    countY += y;
    countZ += z;
  }

  // Block Logic: Check for equilibrium. A body is in equilibrium if the net force in all
  // dimensions is zero.
  if (countX == 0 && countY == 0 && countZ == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}