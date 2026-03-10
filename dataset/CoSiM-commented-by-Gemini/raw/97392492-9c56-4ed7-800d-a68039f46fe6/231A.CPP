/**
 * @file 231A.CPP
 * @brief Solution for a competitive programming problem about team problem-solving.
 * This program calculates the number of problems a three-person team will solve.
 * A problem is solved if at least two out of the three members are confident
 * in their solution (represented by a '1').
 *
 * @algorithm Iterative counting and summation.
 * @time_complexity O(N), where N is the number of problems, as the input is processed in linear time.
 * @space_complexity O(N) to store the confidence levels for all problems.
 */
#include <bits/stdc++.h>
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  int a[2000][3], count[MAX], i, j, k, n, c;
  c = 0; // Final counter for solvable problems
  cin >> n;

  // Initialize helper array for counts.
  for (i = 0; i < n; i++)
    count[i] = 0;

  /**
   * @block
   * @brief Reads the confidence levels for each problem.
   * Input is an N x 3 matrix where N is the number of problems.
   * a[i][j] is 1 if the j-th person is sure about the i-th problem, 0 otherwise.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }

  /**
   * @block
   * @brief Counts the number of confident members for each problem.
   * Iterates through the input matrix and stores the sum of confidence levels
   * for each problem in the `count` array.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      if (a[i][j] == 1)
        count[i]++;
    }
  }

  /**
   * @block
   * @brief Determines the total number of solvable problems.
   * A problem is considered solvable if its confidence count is 2 or greater.
   * The variable `c` accumulates the total count.
   */
  for (i = 0; i < n; i++) {
    if (count[i] >= 2) {
      c = c + 1;
    }
  }

  // Output the final count of solvable problems.
  cout << c;
  return 0;
}
