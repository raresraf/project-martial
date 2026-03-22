/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * @details This program counts how many problems will be solved by a team of three.
 * The condition for solving a problem is that at least two members are sure of
 * the solution. This implementation first stores all inputs in a 2D array
 * before processing them.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the total number of problems, N.
 * 2. Store all N*3 inputs (opinions of 3 members for N problems) in a 2D array.
 * 3. Iterate through each problem (row) in the array.
 * 4. For each problem, count the number of members who are sure (value is 1).
 * 5. An optimization is used: if the count of sure members reaches 2, increment
 *    the final result and immediately break to the next problem, as the condition
 *    is already met.
 * 6. Print the final result.
 *
 * @section complexity
 * - Time Complexity: O(N), as the program reads and then processes N problems.
 * - Space Complexity: O(N), due to storing all inputs in a 2D array of size N x 3.
 */
#include <iostream>
using namespace std;

int main() {
  int arr[1000][3];
  int n;
  cin >> n;

  /**
   * @brief Reads all inputs and populates the 2D array.
   * This block reads N rows, each with 3 integer values, representing the
   * team's confidence for each problem.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= 2; j++) {
      cin >> arr[i][j];
    }
  }

  int count = 0;
  int result = 0;
  /**
   * @brief Processes the stored problems to find the number of solvable ones.
   * @invariant `result` holds the count of problems solvable so far.
   */
  for (int i = 0; i < n; i++) {
    count = 0;
    for (int j = 0; j <= 2; j++) {
      if (arr[i][j] == 1)
        count++;

      /**
       * @brief Optimization: If two members agree, the problem is solved.
       * This check allows the inner loop to terminate early, avoiding an
       * unnecessary check on the third member's opinion if the outcome is
       * already certain.
       */
      if (count == 2) {
        result++;
        break;
      }
    }
  }

  // Output the total count of problems the team will implement.
  cout << result;
  return 0;
}
