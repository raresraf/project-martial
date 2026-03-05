/**
 * @file 231A.CPP
 * @brief Solution to a problem where `n` problems are presented, and for each problem,
 *        three friends (Petya, Vasya, and Tonya) indicate whether they are sure
 *        about the solution (1) or not (0). A problem is implemented if at least
 *        two friends are sure about the solution. The program counts how many
 *        problems will be implemented.
 *
 * Algorithm: Iterative summation and conditional counting.
 *            The algorithm reads the total number of problems `n`. For each problem,
 *            it reads three binary inputs (0 or 1), sums them up, and if the sum
 *            is 2 or more, increments a counter for implemented problems.
 * Time Complexity: O(N), where N is the number of problems.
 *                  The outer loop runs N times, and the inner loop runs a constant
 *                  3 times per problem.
 * Space Complexity: O(1), as only a few variables are used to store counts and sums.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations,
  // which is common practice in competitive programming to meet time limits.
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  // Functional Utility: Initializes `n` for the number of problems, `x` for
  // individual friend's confidence, and `ret` as the counter for implemented problems.
  int n, x, ret = 0;
  // Block Logic: Reads the total number of problems `n` from standard input.
  cin >> n;
  /**
   * Block Logic: Outer loop iterates `n` times, once for each problem.
   * Invariant: `i` is the current problem index, `ret` is the count of
   *            problems implemented so far.
   */
  for (int i = 0; i < n; i++) {
    int ct = 0; // Functional Utility: `ct` temporarily stores the sum of confidence for the current problem.
    /**
     * Block Logic: Inner loop iterates 3 times to read the confidence of three friends.
     * Invariant: `j` is the current friend's index, `x` is their confidence,
     *            `ct` accumulates the confidence sum for the current problem.
     */
    for (int j = 0; j < 3; j++) {
      // Block Logic: Reads a friend's confidence (0 or 1).
      cin >> x;
      // Functional Utility: Adds the friend's confidence to the current problem's total.
      ct += x;
    }
    // Pre-condition: Checks if at least two friends are confident (sum >= 2).
    if (ct >= 2)
      // Block Logic: Increments the counter for implemented problems.
      ret++;
  }
  // Post-condition: Prints the total number of implemented problems followed by a newline.
  cout << ret << endl;
  return 0;
}
