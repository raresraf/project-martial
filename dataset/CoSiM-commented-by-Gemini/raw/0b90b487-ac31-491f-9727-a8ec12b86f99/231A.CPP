/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem by pre-loading all inputs into arrays.
 * @details This program determines how many problems a team of three will solve. It first reads all
 * inputs—the opinions of the three members for `p` problems—into three separate arrays.
 * After all data is stored, it iterates through the arrays, checking if at least two members
 * are confident (value of 1) for each problem. It counts the number of such problems and prints the total.
 *
 * @algorithm Input pre-loading followed by counting.
 * @complexity
 * Time complexity: O(P), where P is the number of problems. The program uses two separate loops, each running P times.
 * Space complexity: O(P), as it uses three arrays of size P to store all inputs before processing. This is less memory-efficient than processing each problem's input one at a time.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // p: the number of problems.
  // a, b, c: arrays to store the opinions of the three team members for each problem.
  // count: a counter for the number of problems to be solved.
  int p, a[1000], b[1000], c[1000], count = 0;

  cin >> p;

  /**
   * @brief Input loop to read and store all opinions.
   * @invariant Iterates `p` times to fill the `a`, `b`, and `c` arrays.
   */
  for (int i = 0; i < p; i++) {
    cin >> a[i] >> b[i] >> c[i];
  }

  /**
   * @brief Processing loop to count solvable problems.
   * @invariant Iterates through the stored opinions for each of the `p` problems.
   */
  for (int i = 0; i < p; i++) {
    /**
     * @brief Checks if at least two members agree.
     * @pre The opinions for the i-th problem are available in a[i], b[i], and c[i].
     * @post The `count` is incremented if any pair of members agrees.
     */
    if (a[i] == 1 && b[i] == 1)
      count++;
    else if (b[i] == 1 && c[i] == 1)
      count++;
    else if (c[i] == 1 && a[i] == 1)
      count++;
  }

  // Print the final count.
  cout << count;

  return 0;
}