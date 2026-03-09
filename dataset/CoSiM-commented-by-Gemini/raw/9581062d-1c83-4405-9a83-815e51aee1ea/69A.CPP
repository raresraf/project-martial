/**
 * @file 69A.CPP
 * @brief Determines if a system of 3D vectors is in equilibrium, for multiple test cases.
 *
 * This program solves the "Young Physicist" problem from Codeforces. It reads
 * a set of 3D vectors and checks if their vector sum is zero. The outer while loop
 * is structured to handle multiple test cases if the input provides them.
 *
 * @algorithm Vector Summation
 * @complexity
 * - Time: O(T * N), where T is the number of test cases and N is the number of
 *   vectors per test case. For a single test case, it is O(N).
 * - Space: O(1) per test case.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // x1, y1, z1: components of a single vector.
  // sum_1, sum_2, sum_3: running totals for x, y, and z components.
  // n: the number of vectors in a single test case.
  int x1, y1, z1, sum_1, sum_2, sum_3, n;

  // The outer loop allows the program to solve multiple test cases.
  while (cin >> n) {
    // Reset sums for the new test case.
    sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;

    /**
     * @brief Block Logic: Reads `n` vectors and accumulates their component sums.
     * Invariant: After each iteration of this inner loop, the `sum_` variables
     *            hold the vector sum of all vectors read so far for the current test case.
     */
    while (n--) {
      cin >> x1 >> y1 >> z1;
      sum_1 += x1;
      sum_2 += y1;
      sum_3 += z1;
    }

    // Pre-condition: Check if the net vector sum is zero (equilibrium).
    if (sum_1 == 0 && sum_2 == 0 && sum_3 == 0)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}