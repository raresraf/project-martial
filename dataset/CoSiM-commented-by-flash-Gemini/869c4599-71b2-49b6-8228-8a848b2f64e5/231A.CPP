/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem involving counting problems solved by a team.
 *
 * Algorithm: Iterative conditional counting.
 * Time Complexity: O(N) where N is the number of problems.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to solve the problem of counting solved problems.
 * Reads the number of problems 'n', then for each problem, reads three participation statuses.
 * A problem is considered solved if at least two participants agree.
 * @return 0 on successful execution.
 */
int main() {
  /**
   * Declares variables:
   * 'n': The total number of problems to be evaluated.
   * 'output': Accumulates the count of problems that are considered solved.
   * Reads the value for 'n' from standard input.
   * Pre-condition: Standard input provides an integer 'n'.
   */
  int n, output = 0; // n cases
  cin >> n;
  /**
   * Declares a 2D array to temporarily store the participation statuses for each problem.
   * Note: While declared, the specific values in this matrix are not directly used
   * for the final output; only the count of '1's per row matters.
   */
  int matx[n][3];
  /**
   * Block Logic: Iterates through each of the 'n' problems.
   * For each problem, it reads the 3 participation statuses and counts how many are '1'.
   * If at least 2 participants agree (count of '1's >= 2), the problem is marked as solved.
   * Invariant: 'output' correctly stores the number of solved problems processed so far.
   */
  for (int i = 0; i < n; i++) {
    /**
     * Initializes a counter for the current problem's agreeing participants.
     */
    int cnt = 0;
    /**
     * Block Logic: Reads the three participation statuses for the current problem.
     * Increments 'cnt' for each participant who agrees (status is '1').
     * Pre-condition: Standard input provides three integers (0 or 1).
     * Invariant: 'cnt' holds the number of '1's encountered for the current problem.
     */
    for (int j = 0; j < 3; j++) {
      cin >> matx[i][j];
      if (matx[i][j] == 1)
        cnt++;
    }
    /**
     * Block Logic: Checks if the current problem meets the solved criteria (at least 2 agreeing participants).
     * If true, increments the total count of solved problems.
     */
    if (cnt >= 2)
      output++;
  }
  /**
   * Prints the total number of problems that meet the solved criteria.
   * Post-condition: The final count of solved problems is output to standard output.
   */
  cout << output;
  return 0;
}
