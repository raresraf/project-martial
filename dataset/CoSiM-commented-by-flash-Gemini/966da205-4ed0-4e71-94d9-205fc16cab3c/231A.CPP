/**
 * @file 231A.CPP
 * @brief Determines the number of problems that a team will implement correctly.
 * * Algorithm: Direct Counting with Threshold.
 * Time Complexity: $O(N \times M)$ where N is the number of problems and M is the number of friends (always 3).
 * Space Complexity: $O(N \times M)$ for storing the input array.
 *
 * This program reads 'n' sets of three integers, representing the opinions of three friends
 * on a problem (1 for confident, 0 for not confident). It counts how many problems
 * have at least two friends confident, as these are the problems the team will implement.
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.
using namespace std;

int main(int argc, char const *argv[]) {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties
   * cin and cout to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int n; // Declares 'n', the number of problems/rows.
  cin >> n; // Reads the value of 'n'.
  int arr[n][3]; // Declares a 2D array to store the input (opinions of 3 friends for 'n' problems).
  int ans = 0; // Initializes 'ans' to count problems solved.

  /**
   * Block Logic: Reads the input data into the `arr` 2D array.
   * Pre-condition: 'n' has been read, specifying the number of rows.
   * Invariant: After each outer iteration, one row of 3 integers is read.
   * Post-condition: The `arr` array is populated with 'n' rows, each with 3 integers.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j]; // Reads each integer (opinion) for the current problem.
    }
  }

  /**
   * Block Logic: Iterates through each problem (row) to determine if it meets the criteria.
   * Pre-condition: `arr` is fully populated with input data.
   * Invariant: `ans` correctly counts problems meeting the criteria for rows processed so far.
   */
  for (int i = 0; i < n; i++) {
    int k = 0; // Local counter 'k' for the number of '1's in the current row.
    // Block Logic: Iterates through the 3 opinions for the current problem (row).
    for (int j = 0; j < 3; j++) {
      // Inline: Checks if the current opinion is '1' (confident).
      if (arr[i][j] == 1) {
        k += 1; // Increments 'k' if the friend is confident.
      }
    }

    // Block Logic: Checks if at least two friends are confident for the current problem.
    if (k >= 2) {
      ans++; // If so, increments the total count of solved problems.
    }
  }
  // Prints the final count of problems that the team will implement correctly.
  cout << ans;

  return 0; // Standard successful exit code for main function.
}