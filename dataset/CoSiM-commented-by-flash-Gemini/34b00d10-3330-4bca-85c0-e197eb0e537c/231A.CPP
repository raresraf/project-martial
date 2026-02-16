/**
 * @file 231A.CPP
 * @brief Solution for a competitive programming problem that counts the number of "solved problems" based on a voting system.
 *
 * Algorithm: Reads 'n' sets of three integer votes (0 or 1). For each set, it counts the number of '1' votes.
 *            If a set has more than one '1' vote, it increments a total counter.
 *            Finally, prints the total count.
 * Time Complexity: O(N) where N is the number of rows (problems). Each row involves a fixed number of operations.
 * Space Complexity: O(N) for storing the 2D array of votes.
 */

#include <iostream>
using namespace std;

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Reads an integer `n`, then processes `n` sets of three
 *                     integer inputs (representing votes). For each set, it
 *                     determines if at least two votes are '1'. It then
 *                     counts how many sets meet this criterion and prints the
 *                     final count.
 *
 * @return 0 on successful execution.
 */
int main() {
  int i; // @brief Loop counter for iterating through rows/problems.
  int j; // @brief Loop counter for iterating through columns/votes.
  int n; // @brief Number of problems (rows of votes).
  int m; // @brief Unused variable.
  int o; // @brief Counter for '1's in the current row/problem.
  int p; // @brief Total counter for problems with more than one '1' vote.
  int a[1001][4]; // @brief 2D array to store the votes for each problem.
  cin >> n;
  // Block Logic: Read 'n' sets of three integer votes into the 2D array `a`.
  // Invariant: After this loop, the array `a` is populated with input data.
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  p = 0; // Functional Utility: Initialize total counter for solved problems.
  // Block Logic: Iterate through each problem (row) to count successful solutions.
  // Invariant: `p` accumulates the count of problems where more than one '1' vote is found.
  for (i = 0; i < n; i++) {
    o = 0; // Functional Utility: Reset '1' vote counter for the current problem.
    // Block Logic: Iterate through the three votes for the current problem.
    // Invariant: `o` counts the number of '1's in `a[i]`.
    for (j = 0; j < 3; j++) {
      if (a[i][j] == 1) { // Inline: Check if the current vote is '1'.
        o++; // Inline: Increment '1' vote counter if the condition is met.
      }
    }
    // Block Logic: Check if the current problem received more than one '1' vote.
    // Invariant: If `o` is greater than 1, the problem is considered "solved".
    if (o > 1) {
      p++; // Functional Utility: Increment the total solved problems counter.
    }
  }
  // Functional Utility: Print the final total count of problems with more than one '1' vote.
  cout << p << endl;
  return 0;
}
