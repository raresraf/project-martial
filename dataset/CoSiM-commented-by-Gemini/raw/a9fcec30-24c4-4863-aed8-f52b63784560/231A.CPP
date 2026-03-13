/**
 * @file 231A.CPP
 * @brief Solution for CodeForces Problem 231A - "Team".
 *
 * @details
 * The problem is to count the number of problems a team will implement.
 * The team decides to solve a problem if at least two of the three members are sure.
 *
 * @algorithm Two-pass array-based check
 * 1. Read the total number of problems, `n`.
 * 2. Read all `n` sets of three opinions (0 or 1) into a 2D array.
 * 3. Initialize a final counter, `l`, to 0.
 * 4. Iterate through each row (problem) of the stored inputs:
 *    a. Use an explicit boolean condition to check if at least two members are sure.
 *       `(a[i][0] && a[i][1]) || (a[i][1] && a[i][2]) || (a[i][0] && a[i][2])`
 *    b. If the condition is true, increment the `l` counter.
 * 5. Print the final count `l`.
 *
 * This approach is functionally correct but less direct than simply summing the
 * three opinions and checking if the sum is >= 2. It also uses O(N) memory
 * to store all inputs before processing.
 *
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(N) due to the Variable Length Array `a[n][3]`.
 */
#include <iostream>
using namespace std;
int main() {
  int n;
  int l = 0; // Counter for implementable problems.
  cin >> n;
  
  // Using a Variable Length Array (VLA), a non-standard C++ extension.
  int a[n][3];

  /**
   * @brief First pass: read all inputs into the 2D array.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }

  /**
   * @brief Second pass: iterate through inputs to count valid problems.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief Check if at least two members agree.
     * This is equivalent to checking if `a[i][0] + a[i][1] + a[i][2] >= 2`.
     */
    if ((a[i][0] == 1 && a[i][1] == 1) || (a[i][2] == 1 && a[i][1] == 1) ||
        (a[i][0] == 1 && a[i][2] == 1)) {
      l++;
    }
  }

  cout << l;
  return 0;
}
