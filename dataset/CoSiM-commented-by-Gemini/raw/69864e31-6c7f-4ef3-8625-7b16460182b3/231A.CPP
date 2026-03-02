#include <bits/stdc++.h>
using namespace std;

/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem from Codeforces (Problem 231A).
 * @details This program calculates the number of problems a three-person team will solve,
 * based on whether at least two members are confident in the solution.
 *
 * Algorithm:
 * This solution uses a two-pass approach. First, it reads all the input data and
 * stores it in a 2D array. Then, it processes the array to get the result.
 * 1. The number of problems, `n`, is read. A 2D array `a[n][3]` is used to store
 *    the confidence votes for all problems.
 * 2. The program reads all `n * 3` votes and populates the array.
 * 3. A second loop iterates through each row of the array (each problem). It sums
 *    the votes for that problem.
 * 4. If the sum is 2 or greater, a counter for solved problems is incremented.
 * 5. Finally, the total count is printed.
 * Note: This uses a Variable Length Array (VLA), a non-standard C++ extension.
 *
 * Time Complexity: O(n), for the two separate passes over the `n` problems.
 * Space Complexity: O(n), to store the `n x 3` matrix of votes.
 */
int main() {
  int n;
  cin >> n;
  int a[n][3], sum, count = 0;

  /**
   * @brief First pass: Read all inputs into a 2D array.
   * Each row `i` represents a problem, and the three columns are the votes.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }

  /**
   * @brief Second pass: Process the stored data to find the answer.
   */
  for (int i = 0; i < n; i++) {
    sum = 0;
    // Sum the votes for the current problem.
    for (int j = 0; j < 3; j++) {
      sum += a[i][j];
    }
    // If two or more members are sure, increment the solution count.
    if (sum > 1)
      count += 1;
  }
  cout << count;
  // your code goes here

  return 0;
}
