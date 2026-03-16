/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 *
 * @details
 * This program calculates the number of problems a team of three will solve. A problem
 * is solved if at least two of the three members are confident about the solution.
 *
 * This implementation uses a two-stage approach:
 * 1. Data Ingestion: It first reads all `n` problems and their corresponding three
 *    confidence values (0 or 1) into a 2D array.
 * 2. Processing: It then iterates through the stored data, explicitly counting the
 *    number of confident members for each problem. If the count is 2 or more, a
 *    result counter is incremented.
 *
 * This approach is functionally correct but less space-efficient than solutions that
 * process each problem's data immediately upon reading it, as it requires O(N) space
 * to store all inputs.
 *
 * @note The code uses a Variable Length Array (`int arr[n][3]`), which is a
 * non-standard C++ extension.
 *
 * Algorithm:
 * 1. Read `n`, the number of problems.
 * 2. Store all `n x 3` inputs in a 2D array.
 * 3. Iterate through each row of the array, counting the 1s.
 * 4. If the count for a row is >= 2, increment the final result.
 * 5. Print the result.
 *
 * Time Complexity: O(n), for iterating through the problems.
 * Space Complexity: O(n), to store the `n x 3` input array.
 */
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n;
  cin >> n;

  /**
   * @brief Data Ingestion Phase.
   * @details Reads all problem data into a 2D Variable Length Array before processing.
   * Each row `i` represents a problem, and the columns `j` represent the team members.
   */
  int arr[n][3];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  int res = 0;
  /**
   * @brief Processing Phase.
   * @details Iterates through the stored data to find the number of solvable problems.
   */
  for (int i = 0; i < n; i++) {
    int counter = 0;
    // Inner loop explicitly counts confident members (value of 1) for problem `i`.
    for (int j = 0; j < 3; j++) {
      if (arr[i][j] == 1)
        counter++;
    }
    // If two or more members are confident, the problem is counted.
    if (counter >= 2) {
      res++;
      counter = 0; // This reset is redundant as `counter` is re-declared in the next iteration.
    }
  }

  cout << res;
  return 0;
}