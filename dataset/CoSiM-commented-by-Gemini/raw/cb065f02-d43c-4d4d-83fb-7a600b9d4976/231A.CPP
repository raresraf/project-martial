/**
 * @file 231A.CPP
 * @brief A logically convoluted solution for Codeforces problem 231A, "Team".
 * @details This program counts the number of problems a team will implement, based
 * on the condition that at least two of the three members are confident. This
 * implementation uses an unusual and verbose conditional structure.
 *
 * @section Algorithm
 * The program iterates through each problem and explicitly checks pairs of team
 * members to see if they are confident.
 *
 * 1. Read the number of problems, `n`.
 * 2. Initialize a counter `x` for solvable problems to 0.
 * 3. Loop `n` times, once for each problem.
 * 4. Inside the loop, read the three opinions into a boolean array `arr`.
 * 5. An `if-else if` chain checks for combinations of two confident members:
 *    - `arr[0]` and `arr[2]`
 *    - `arr[1]` and `arr[0]`
 *    - `arr[1]` and `arr[2]`
 * 6. If any of these conditions are met, the counter `x` is incremented. The chain
 *    of `else if` ensures that `x` is only incremented once per problem.
 *
 * This logic is unnecessarily complex. A much simpler approach would be to sum the
 * boolean values and check if the sum is greater than 1. Although convoluted, the
 * code produces the correct result as the case of all three members being confident
 * is caught by the first `if` condition.
 *
 * @subsection Complexity
 * - Time Complexity: O(n), for iterating through `n` problems.
 * - Space Complexity: O(1), using a fixed-size array.
 */
#include <iostream>

using namespace std;

int main() {
  // n: the number of problems. x: the count of problems to be solved.
  int n = 0, x = 0;
  // arr: stores the three opinions (1 or 0) as booleans.
  bool arr[3];

  cin >> n;

  /**
   * @brief This loop iterates through each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the opinions of the three members for the current problem.
    for (int a = 0; a < 3; a++) {
      cin >> arr[a];
    }
    /**
     * @brief This block of conditions checks if at least two members are sure.
     * It explicitly checks pairs of members. This is functionally correct for
     * this problem but is a verbose way to achieve the result.
     */
    if (arr[0] && arr[2] == true) {
      x++;
    } else if (arr[1] && arr[0] == true) {
      x++;
    } else if (arr[1] && arr[2] == true) {
      x++;
    }
  }
  // Print the final count of problems to be solved.
  cout << x;
}
